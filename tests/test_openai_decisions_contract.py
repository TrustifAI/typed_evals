"""Real OpenAI SDK contracts; only HTTP is mocked, never billable requests."""

import copy
import json

import httpx2
import pytest

openai = pytest.importorskip("openai", minversion="3.26.0")

from typed_evals import (  # noqa: E402
    EvaluationSample,
    Evaluator,
    Metric,
    OpenAIDecisionsBackend,
    ToolAccuracy,
    ToolProposal,
)
from typed_evals.backends.openai_decisions import (  # noqa: E402
    compile_questions,
    flatten_usage,
    serialize_state,
)
from typed_evals.errors import DecisionRefusalError, InvalidAnswerError  # noqa: E402
from typed_evals.metrics.base import EVALUATION_POLICY  # noqa: E402


def sdk_client(handler, **options):
    return openai.AsyncOpenAI(
        api_key="test-only-not-real",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
        max_retries=0,
        **options,
    )


def wire_response(answers, model="gpt-6-luna-observed"):
    return {
        "model": model,
        "answers": answers,
        "usage": {
            "input_tokens": 25,
            "output_tokens": 0,
            "total_tokens": 25,
            "input_tokens_details": {"cached_tokens": 3, "cache_write_tokens": 2},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


def panel():
    return [
        Metric(
            name="positive",
            instructions="Is this acceptable? 完整 " + "rubric " * 2000,
            criteria={"true": "Positive acceptable evidence ✓", "false": "Negative evidence ✗"},
            required_fields=("input", "response", "proposed_tool_call"),
            pass_definition="Acceptable",
        ),
        Metric(
            name="category",
            kind="choice",
            instructions="Select category",
            criteria={
                "pass_a": "First acceptable",
                "fail": "Unacceptable",
                "pass_b": "Second acceptable",
            },
            pass_options=("pass_a", "pass_b"),
            pass_definition="Either acceptable category",
        ),
        Metric(
            name="coverage",
            kind="score",
            instructions="Rate coverage",
            criteria=("Repeated text", "Repeated text", "Full coverage"),
            pass_definition="Coverage adequate",
        ),
    ]


def good_answers():
    return [
        {"name": "positive", "type": "predicate", "probability": 0.9},
        {
            "name": "category",
            "type": "choice",
            "choice": "fail",
            "confidence": 0.13,
            "probabilities": [
                {"value": "pass_a", "probability": 0.35},
                {"value": "pass_b", "probability": 0.25},
                {"value": "fail", "probability": 0.4},
            ],
        },
        {
            "name": "coverage",
            "type": "score",
            "score": 1.5,
            "confidence": 0.27,
            "probabilities": [
                {"label": "0", "value": 0, "probability": 0.1},
                {"label": "1", "value": 1, "probability": 0.3},
                {"label": "2", "value": 2, "probability": 0.6},
            ],
        },
    ]


@pytest.fixture
def tool_sample():
    return EvaluationSample(
        input="cafe",
        response="30 days",
        contexts=("unused secret context",),
        reference="unused reference",
        trace=({"name": "unused_tool", "output": "unused trace"},),
        proposed_tool_call=ToolProposal(
            name="lookup", arguments={"query": "cafe", "nested": {"id": [42]}}
        ),
        id="not-evidence",
        group_id="not-evidence",
        metadata={"secret": "not-evidence"},
    )


async def test_all_questions_one_request_complete_rubric_filtered_evidence(tool_sample):
    observed = []

    def handler(request):
        assert request.method == "POST"
        assert request.url.path == "/v1/decisions"
        observed.append(json.loads(request.content))
        # Name association works independently of response order.
        return httpx2.Response(200, json=wire_response(list(reversed(good_answers()))))

    metrics = panel()
    async with sdk_client(handler) as client:
        row = await Evaluator(metrics, backend=OpenAIDecisionsBackend(client=client)).aevaluate_one(
            tool_sample
        )
    assert len(observed) == 1
    body = observed[0]
    assert set(body) == {"model", "input", "questions"}
    assert body["model"] == "gpt-6-luna"
    assert json.loads(body["input"]) == tool_sample.state(
        {"input", "response", "proposed_tool_call"}
    )
    assert "cafe" in body["input"]
    questions = body["questions"]
    assert [q["name"] for q in questions] == ["positive", "category", "coverage"]
    predicate = questions[0]
    assert predicate["type"] == "predicate"
    assert "criteria" not in predicate
    assert (
        metrics[0].instructions in json.loads(predicate["instructions"].split("\n")[0])["question"]
    )
    assert EVALUATION_POLICY in predicate["instructions"]
    assert metrics[0].criteria["true"] in predicate["instructions"]
    assert metrics[0].criteria["false"] in predicate["instructions"]
    assert "positive (true) condition" in predicate["instructions"]
    assert [option["value"] for option in questions[1]["choices"]] == ["fail", "pass_a", "pass_b"]
    assert questions[2]["levels"] == [
        {"label": "0", "description": "Repeated text"},
        {"label": "1", "description": "Repeated text"},
        {"label": "2", "description": "Full coverage"},
    ]
    assert row.model == "gpt-6-luna-observed"
    assert row.metrics["positive"].raw_score == 0.90
    assert row.metrics["positive"].confidence is None
    assert row.metrics["category"].raw_score == 0.60
    assert row.metrics["category"].selected_choice == "fail"
    assert row.metrics["category"].confidence == 0.13
    assert row.metrics["coverage"].raw_score == 0.75
    assert row.metrics["coverage"].confidence == 0.27
    assert row.metrics["coverage"].probabilities == {"0": 0.1, "1": 0.3, "2": 0.6}
    assert all(m.calibrated_probability is None for m in row.metrics.values())
    assert row.usage == {
        "input_tokens": 25,
        "output_tokens": 0,
        "total_tokens": 25,
        "input_tokens_details.cached_tokens": 3,
        "input_tokens_details.cache_write_tokens": 2,
        "output_tokens_details.reasoning_tokens": 0,
    }


async def test_true_false_options_remain_strings(sample):
    observed = []

    def handler(request):
        observed.append(json.loads(request.content))
        return httpx2.Response(
            200,
            json=wire_response(
                [
                    {
                        "name": "tool_accuracy",
                        "type": "choice",
                        "choice": "true",
                        "confidence": 0.2,
                        "probabilities": [
                            {"value": "true", "probability": 0.9},
                            {"value": "false", "probability": 0.1},
                        ],
                    }
                ]
            ),
        )

    sample = sample.model_copy(update={"proposed_tool_call": ToolProposal(name="lookup")})
    async with sdk_client(handler) as client:
        row = await Evaluator(
            [ToolAccuracy()], backend=OpenAIDecisionsBackend(client=client)
        ).aevaluate_one(sample)
    choices = observed[0]["questions"][0]["choices"]
    assert {option["value"] for option in choices} == {"true", "false"}
    assert all(type(option["value"]) is str for option in choices)
    assert row.metrics["tool_accuracy"].raw_score == 0.9


def malformed_answers():
    cases = []
    for value in [True, False, "0.9", None, float("nan"), float("inf"), -0.1, 1.1, [], 10**1000]:
        cases.append(("positive", {"probability": value}))
    cases.extend(("positive", {"type": value}) for value in ["noul", "choice", None, True])
    for field in ["confidence", "score"]:
        for value in [True, "1", None, float("nan"), float("inf"), -1, 3, 10**1000]:
            cases.append(("coverage", {field: value}))
    for value in [True, "0.13", None, float("nan"), 1.2, -0.1]:
        cases.append(("category", {"confidence": value}))
    for value in [True, 1, None, "unknown"]:
        cases.append(("category", {"choice": value}))
    for value in [True, False, "0.35", None, float("nan"), float("inf"), -0.1, 1.1, 10**1000]:
        items = copy.deepcopy(good_answers()[1]["probabilities"])
        items[0]["probability"] = value
        cases.append(("category", {"probabilities": items}))
    for value in [True, 1, None, [], "unknown"]:
        items = copy.deepcopy(good_answers()[1]["probabilities"])
        items[0]["value"] = value
        cases.append(("category", {"probabilities": items}))
    for value in [True, 0.0, "0", -1, 3, None]:
        items = copy.deepcopy(good_answers()[2]["probabilities"])
        items[0]["value"] = value
        cases.append(("coverage", {"probabilities": items}))
    for value in [0, True, "1", None]:
        items = copy.deepcopy(good_answers()[2]["probabilities"])
        items[0]["label"] = value
        cases.append(("coverage", {"probabilities": items}))
    for index, name in [(1, "category"), (2, "coverage")]:
        original = good_answers()[index]["probabilities"]
        for items in [
            None,
            {},
            original[:-1],
            original + [original[0]],
            [original[0]] * 3,
            [None],
            [],
        ]:
            cases.append((name, {"probabilities": items}))
    cases.append(("category", {"choice": "pass_b"}))
    cases.append(("coverage", {"score": 0.5}))
    cases.append(
        (
            "category",
            {
                "probabilities": [
                    {"value": v, "probability": 0.1} for v in ["pass_a", "pass_b", "fail"]
                ]
            },
        )
    )
    return cases


@pytest.mark.parametrize("name,patch", malformed_answers())
async def test_malformed_wire_values_never_coerced_and_preserve_siblings(name, patch, tool_sample):
    answers = good_answers()
    next(answer for answer in answers if answer["name"] == name).update(patch)

    def handler(request):
        # content supports deliberately nonfinite fixtures rejected before normalization.
        return httpx2.Response(
            200,
            content=json.dumps(wire_response(answers)),
            headers={"content-type": "application/json"},
        )

    async with sdk_client(handler) as client:
        backend = OpenAIDecisionsBackend(client=client)
        row = await Evaluator(panel(), backend=backend, errors="record").aevaluate_one(tool_sample)
        with pytest.raises(InvalidAnswerError):
            await Evaluator(panel(), backend=backend).aevaluate_one(tool_sample)
    assert row.metrics[name].status == "error"
    assert row.metrics[name].score is None
    assert row.metrics[name].passed is None
    assert all(metric.status == "ok" for other, metric in row.metrics.items() if other != name)
    assert row.passed is None


@pytest.mark.parametrize(
    "answers",
    [
        None,
        {},
        [None],
        [],
        good_answers()[:-1],
        good_answers() + [good_answers()[0]],
        [{**answer, "name": None} for answer in good_answers()],
        [{**answer, "name": True} for answer in good_answers()],
        [{**answer, "name": []} for answer in good_answers()],
        [{**answer, "name": "unexpected"} for answer in good_answers()],
    ],
)
async def test_untrustworthy_response_association_is_request_failure(answers, tool_sample):
    async with sdk_client(lambda r: httpx2.Response(200, json=wire_response(answers))) as client:
        backend = OpenAIDecisionsBackend(client=client)
        with pytest.raises(InvalidAnswerError):
            await Evaluator(panel(), backend=backend).aevaluate_one(tool_sample)
        row = await Evaluator(panel(), backend=backend, errors="record").aevaluate_one(tool_sample)
    assert all(metric.status == "error" for metric in row.metrics.values())
    assert row.passed is None


@pytest.mark.parametrize("size", [2, 255, 256])
def test_choice_count_constraints(size):
    metric = Metric(
        name="category",
        kind="choice",
        instructions="Choose",
        criteria={str(i): f"option {i}" for i in range(size)},
        pass_options=("0",),
        pass_definition="zero",
    )
    if size <= 255:
        assert len(compile_questions({metric.name: metric.question()})[0]["choices"]) == size
    else:
        with pytest.raises(ValueError, match="2 and 255"):
            compile_questions({metric.name: metric.question()})


def test_serialization_and_question_order_are_deterministic():
    assert serialize_state({"response": "✓", "input": "退款"}) == serialize_state(
        {"input": "退款", "response": "✓"}
    )
    metric = panel()[1]
    other = metric.model_copy(update={"criteria": dict(reversed(list(metric.criteria.items())))})
    assert compile_questions({metric.name: metric.question()}) == compile_questions(
        {other.name: other.question()}
    )


@pytest.mark.parametrize(
    "usage",
    [
        None,
        [],
        {"input_tokens": True},
        {"input_tokens": "25"},
        {"nested": {"bad": float("nan")}},
        {"input_tokens": -1},
    ],
)
def test_usage_rejects_nonscalar_or_coerced_counters(usage):
    with pytest.raises(InvalidAnswerError):
        flatten_usage(usage)


def test_flat_usage_supports_optional_and_nested_detail_counters():
    assert flatten_usage(
        {"input_tokens": 3, "new_details": {"new_tokens": 2, "missing": None}}
    ) == {"input_tokens": 3, "new_details.new_tokens": 2, "new_details.missing": None}


@pytest.mark.parametrize("errors", ["raise", "record"])
async def test_mixed_success_and_refusal_preserves_successful_siblings(errors, tool_sample):
    answers = good_answers()
    answers[1] = {"name": "category", "type": "refusal"}
    async with sdk_client(lambda r: httpx2.Response(200, json=wire_response(answers))) as client:
        evaluator = Evaluator(panel(), backend=OpenAIDecisionsBackend(client=client), errors=errors)
        if errors == "raise":
            with pytest.raises(DecisionRefusalError):
                await evaluator.aevaluate_one(tool_sample)
            return
        row = await evaluator.aevaluate_one(tool_sample)
    assert row.metrics["positive"].raw_score == 0.9
    assert row.metrics["coverage"].raw_score == 0.75
    refused = row.metrics["category"]
    assert refused.status == "error"
    assert refused.score is refused.raw_score is refused.passed is None
    assert row.passed is None
    assert "DecisionRefusalError" in refused.message


async def test_high_cardinality_distribution_cannot_be_all_zero(sample):
    metric = Metric(
        name="many_options",
        kind="choice",
        instructions="Choose",
        pass_definition="zero",
        criteria={str(i): f"option {i}" for i in range(255)},
        pass_options=("0",),
    )
    answer = {
        "type": "choice",
        "name": metric.name,
        "choice": "0",
        "confidence": 0.9,
        "probabilities": [{"value": str(i), "probability": 0} for i in range(255)],
    }
    async with sdk_client(lambda r: httpx2.Response(200, json=wire_response([answer]))) as client:
        with pytest.raises(InvalidAnswerError):
            await Evaluator([metric], backend=OpenAIDecisionsBackend(client=client)).aevaluate_one(
                sample
            )


@pytest.mark.parametrize(
    "content",
    ["not JSON", "null", "[]", '{"model":true,"answers":[]}', '{"model":"","answers":[]}'],
)
async def test_invalid_top_level_wire_is_request_failure(content, sample):
    async with sdk_client(
        lambda r: httpx2.Response(
            200, content=content, headers={"content-type": "application/json"}
        )
    ) as client:
        with pytest.raises(InvalidAnswerError):
            await Evaluator(backend=OpenAIDecisionsBackend(client=client)).aevaluate_one(sample)


async def test_duplicate_json_fields_cannot_erase_malformed_values(sample):
    content = '{"model":"gpt-6-luna","usage":{},"answers":[{"name":"answer_relevancy","type":"predicate","probability":true,"probability":0.9}]}'
    async with sdk_client(
        lambda r: httpx2.Response(
            200, content=content, headers={"content-type": "application/json"}
        )
    ) as client:
        backend = OpenAIDecisionsBackend(client=client)
        with pytest.raises(InvalidAnswerError):
            await Evaluator(backend=backend).aevaluate_one(sample)
        result = await Evaluator(backend=backend, errors="record").aevaluate_one(sample)
    assert result.metrics["answer_relevancy"].status == "error"
    assert result.passed is None
