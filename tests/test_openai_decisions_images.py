"""Real SDK image request contracts, with only the HTTP transport mocked."""

import copy
import json

import pytest

openai = pytest.importorskip("openai", minversion="3.26.0")
httpx2 = pytest.importorskip("httpx2")

from test_image_backends import image_metric, image_sample  # noqa: E402
from test_openai_decisions_contract import (  # noqa: E402
    good_answers,
    panel,
    sdk_client,
    wire_response,
)
from test_openai_decisions_contract import tool_sample as tool_sample  # noqa: E402

from typed_evals import Evaluator, ImageInput, OpenAIDecisionsBackend  # noqa: E402
from typed_evals.backends.openai_decisions import (  # noqa: E402
    compile_input,
    serialize_state,
)
from typed_evals.metrics.base import EVALUATION_POLICY  # noqa: E402


async def test_native_images_and_filtered_text_share_one_request(tool_sample):
    images = (
        ImageInput.from_bytes(b"first image", mime_type="image/png", detail="high", label="商品"),
        ImageInput.from_bytes(b"second image", mime_type="image/jpeg", detail="original"),
    )
    sample = tool_sample.model_copy(update={"images": images})
    metrics = panel()
    metrics[0] = metrics[0].model_copy(
        update={"required_fields": (*metrics[0].required_fields, "images")}
    )
    observed = []

    def handler(request):
        assert request.url.path == "/v1/decisions"
        observed.append(json.loads(request.content))
        return httpx2.Response(200, json=wire_response(good_answers()))

    async with sdk_client(handler) as client:
        row = await Evaluator(metrics, backend=OpenAIDecisionsBackend(client=client)).aevaluate_one(
            sample
        )
    assert len(observed) == 1
    body = observed[0]
    assert len(body["questions"]) == 3
    assert len(body["input"]) == 1
    message = body["input"][0]
    assert message["role"] == "user"
    text, *parts = message["content"]
    assert text["type"] == "input_text"
    state = json.loads(text["text"])
    assert state == {
        **sample.state({"input", "response", "proposed_tool_call"}),
        "images": [
            {"index": 0, "label": "商品", "detail": "high"},
            {"index": 1, "label": None, "detail": "original"},
        ],
    }
    assert parts == [
        {"type": "input_image", "image_url": image.data_url, "detail": image.detail}
        for image in images
    ]
    assert all(image.data_url not in text["text"] for image in images)
    assert "unused secret" not in json.dumps(body)
    assert EVALUATION_POLICY in body["questions"][0]["instructions"]
    assert row.metrics["positive"].raw_score == 0.9
    assert row.metrics["category"].raw_score == 0.6
    assert row.metrics["coverage"].raw_score == 0.75
    assert all(image.data_url not in row.model_dump_json() for image in images)


async def test_images_not_selected_by_metrics_keep_the_text_wire_contract(sample):
    sample = sample.model_copy(update={"images": image_sample().images})
    observed = []

    def handler(request):
        observed.append(json.loads(request.content))
        return httpx2.Response(
            200,
            json=wire_response(
                [{"name": "answer_relevancy", "type": "predicate", "probability": 0.8}]
            ),
        )

    async with sdk_client(handler) as client:
        await Evaluator(backend=OpenAIDecisionsBackend(client=client)).aevaluate_one(sample)
    assert observed[0]["input"] == serialize_state(sample.state({"input", "response"}))


@pytest.mark.parametrize("size", [128, 129])
async def test_decisions_image_count_boundary_before_any_http_request(size):
    sample = image_sample().model_copy(update={"images": image_sample().images * size})
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        return httpx2.Response(
            200,
            json=wire_response(
                [{"name": "visual_grounding", "type": "predicate", "probability": 0.9}]
            ),
        )

    async with sdk_client(handler) as client:
        evaluator = Evaluator([image_metric()], backend=OpenAIDecisionsBackend(client=client))
        if size == 129:
            with pytest.raises(ValueError, match="at most 128 images"):
                await evaluator.aevaluate_one(sample)
            assert not calls
        else:
            assert (await evaluator.aevaluate_one(sample)).passed
            assert len(calls[0]["input"][0]["content"]) == 129


def test_image_compilation_preserves_input_and_deterministic_text():
    state = image_sample().state({"input", "response", "images"})
    snapshot = copy.deepcopy(state)
    assert compile_input(state) == compile_input(dict(reversed(list(state.items()))))
    assert state == snapshot
    assert compile_input({"input": "退款", "response": "✓"}) == serialize_state(
        {"input": "退款", "response": "✓"}
    )


async def test_oversized_image_batch_is_preflighted_before_any_request():
    calls = []

    def handler(request):
        calls.append(request)
        pytest.fail("No sample should be sent from a batch with invalid image counts")

    valid = image_sample()
    oversized = valid.model_copy(update={"images": valid.images * 129})
    async with sdk_client(handler) as client:
        evaluator = Evaluator([image_metric()], backend=OpenAIDecisionsBackend(client=client))
        with pytest.raises(ValueError, match="at most 128 images"):
            await evaluator.aevaluate([valid, oversized])
    assert not calls


@pytest.mark.parametrize(
    "images", [{"data_url": "bad"}, [{"data_url": "https://invalid/image.png"}]]
)
def test_low_level_image_compilation_revalidates_neutral_evidence(images):
    with pytest.raises(ValueError):
        compile_input({"images": images})


async def test_image_only_metric_sends_native_visual_evidence(sample):
    sample = sample.model_copy(update={"images": image_sample().images})
    metric = image_metric().model_copy(update={"required_fields": ("images",)})
    observed = []

    def handler(request):
        observed.append(json.loads(request.content))
        return httpx2.Response(
            200,
            json=wire_response([{"name": metric.name, "type": "predicate", "probability": 0.9}]),
        )

    async with sdk_client(handler) as client:
        row = await Evaluator(
            [metric], backend=OpenAIDecisionsBackend(client=client)
        ).aevaluate_one(sample)
    assert row.passed
    content = observed[0]["input"][0]["content"]
    assert set(json.loads(content[0]["text"])) == {"images"}
    assert content[1]["type"] == "input_image"
