import json
from contextlib import asynccontextmanager

import pytest
from conftest import FakeBackend, binary_answer, calibrated_rows

from typed_evals.backends import JevBackend, OpenAIDecisionsBackend
from typed_evals.cli import main


@pytest.mark.parametrize("override,threshold,exit_code", [(None, 0.8, 1), (0.5, 0.5, 0)])
def test_cli_preserves_metric_threshold_unless_explicitly_overridden(
    override, threshold, exit_code, tmp_path, monkeypatch
):
    import typed_evals.cli as cli

    backend = FakeBackend(
        lambda state, questions: {
            name: binary_answer(question, 0.79) for name, question in questions.items()
        }
    )
    monkeypatch.setattr(cli, "JevBackend", lambda **kwargs: backend)
    dataset, output = tmp_path / "proposals.json", tmp_path / "report.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "input": "Read ticket T-42",
                    "response": "",
                    "contexts": ["read_ticket(ticket_id) reads a ticket; requested ID is T-42."],
                    "proposed_tool_call": {
                        "name": "read_ticket",
                        "arguments": {"ticket_id": "T-42"},
                    },
                }
            ]
        )
    )
    args = [
        "evaluate",
        str(dataset),
        "--metrics",
        "tool_accuracy",
        "--fail-on-failure",
        "--output",
        str(output),
    ]
    if override is not None:
        args.extend(["--threshold", str(override)])
    assert main(args) == exit_code
    metric = json.loads(output.read_text())["results"][0]["metrics"]["tool_accuracy"]
    assert metric["threshold"] == threshold
    assert metric["passed"] is (exit_code == 0)


@pytest.mark.parametrize(
    "algorithm,cutoff,expected",
    [
        ("auto", 2000, "venn_abers"),
        ("auto", 20, "isotonic"),
        ("venn_abers", 20, "venn_abers"),
        ("isotonic", 2000, "isotonic"),
    ],
)
def test_cli_calibration_algorithm_and_cutoff_reach_saved_predictor(
    algorithm, cutoff, expected, scoring_backend, tmp_path, monkeypatch, capsys
):
    import typed_evals.cli as cli

    monkeypatch.setattr(cli, "JevBackend", lambda **kwargs: scoring_backend)
    train, validation, output = (tmp_path / name for name in ("train.json", "val.json", "fit.json"))
    train.write_text(json.dumps([row.model_dump(mode="json") for row in calibrated_rows()]))
    validation.write_text(
        json.dumps([row.model_dump(mode="json") for row in calibrated_rows("val")])
    )
    assert (
        main(
            [
                "calibrate",
                str(train),
                "--validation-data",
                str(validation),
                "--output",
                str(output),
                "--min-samples",
                "20",
                "--min-validation-samples",
                "10",
                "--algorithm",
                algorithm,
                "--isotonic-min-samples",
                str(cutoff),
            ]
        )
        == 0
    )
    artifact = json.loads(output.read_text())
    assert artifact["curves"]["answer_relevancy"]["algorithm"] == expected
    assert (
        json.loads(capsys.readouterr().out)["metrics"]["answer_relevancy"]["algorithm"] == expected
    )

    production, report_path = tmp_path / "production.json", tmp_path / "report.json"
    production.write_text('[{"input":"production-only", "response":"0.8"}]')
    assert (
        main(
            [
                "evaluate",
                str(production),
                "--calibration",
                str(output),
                "--threshold",
                "0.7",
                "--fail-on-failure",
                "--output",
                str(report_path),
            ]
        )
        == 1
    )
    report = json.loads(report_path.read_text())
    metric = report["results"][0]["metrics"]["answer_relevancy"]
    assert report["calibrated"] is True
    assert metric["raw_score"] == 0.8
    assert 0.5 < metric["score"] < 0.7
    assert metric["calibrated_probability"] == metric["score"]
    assert metric["calibration_target"] == "metric_pass"
    assert report["results"][0]["passed"] is False
    summary = json.loads(capsys.readouterr().out)["answer_relevancy"]
    assert summary["mean_raw_score"] == 0.8
    assert summary["mean_score"] == metric["score"]


@pytest.mark.parametrize("command", ["evaluate", "calibrate"])
@pytest.mark.parametrize(
    "backend_name,backend_class,default_model",
    [
        ("jev", JevBackend, "jev-1.14.0"),
        ("openai-decisions", OpenAIDecisionsBackend, "gpt-6-luna"),
    ],
)
@pytest.mark.parametrize("model", [None, "explicit-model"])
def test_cli_selected_backend_and_model_produce_real_reports_and_calibration(
    command,
    backend_name,
    backend_class,
    default_model,
    model,
    tmp_path,
    monkeypatch,
    scoring_backend,
    capsys,
):
    import typed_evals.cli as cli

    selected = []
    expected_model = model or default_model
    scoring_backend.actual_model = expected_model

    @asynccontextmanager
    async def offline_session(instance):
        selected.append(instance)
        async with scoring_backend.session() as session:
            yield session

    # Keep construction, provenance, pipeline, dataset parsing, and persistence real.
    # Replace only the provider session so the test makes no paid requests.
    monkeypatch.setattr(backend_class, "session", offline_session)
    other_class = OpenAIDecisionsBackend if backend_name == "jev" else JevBackend

    def unexpected_backend(**kwargs):
        raise AssertionError("CLI selected the wrong backend")

    monkeypatch.setattr(cli, other_class.__name__, unexpected_backend)
    payload = (
        [row.model_dump(mode="json") for row in calibrated_rows()]
        if command == "calibrate"
        else [{"input": "Question", "response": "0.8", "id": "production"}]
    )
    dataset = tmp_path / "rows.json"
    dataset.write_text(json.dumps(payload))
    output = tmp_path / "output.json"
    args = [command, str(dataset), "--backend", backend_name, "--output", str(output)]
    if model is not None:
        args.extend(["--model", model])
    if command == "calibrate":
        validation = tmp_path / "validation.json"
        validation.write_text(
            json.dumps([row.model_dump(mode="json") for row in calibrated_rows("validation")])
        )
        args.extend(
            [
                "--validation-data",
                str(validation),
                "--min-samples",
                "20",
                "--min-validation-samples",
                "10",
            ]
        )
    assert main(args) == 0
    assert selected
    assert all(
        isinstance(instance, backend_class) and instance.model == expected_model
        for instance in selected
    )
    artifact = json.loads(output.read_text())
    if command == "evaluate":
        row = artifact["results"][0]
        assert row["model"] == expected_model
        assert row["sample_id"] == "production"
        assert row["metrics"]["answer_relevancy"]["score"] == 0.8
        assert row["passed"] is True
        assert json.loads(capsys.readouterr().out)["answer_relevancy"]["passed"] == 1
    else:
        assert artifact["requested_model"] == artifact["observed_model"] == expected_model
        assert artifact["curves"]["answer_relevancy"]["n_samples"] == 40
        assert (
            json.loads(capsys.readouterr().out)["metrics"]["answer_relevancy"]["validation_samples"]
            == 40
        )
    assert scoring_backend.sessions == scoring_backend.closed == len(selected)


@pytest.mark.parametrize(
    "outcome,expected_status,expected_exit",
    [
        ("pass", "ok", 0),
        ("fail", "ok", 1),
        ("skip", "skipped", 1),
        ("error", "error", 1),
        ("empty", None, 1),
    ],
)
def test_cli_ci_exit_status_and_saved_report_agree(
    outcome, expected_status, expected_exit, tmp_path, monkeypatch, capsys
):
    import typed_evals.cli as cli

    def answer(state, questions):
        if outcome == "error":
            raise RuntimeError("private provider detail")
        return {"faithfulness": {"type": "noul", "noul": 0.8 if outcome == "pass" else 0.2}}

    backend = FakeBackend(answer)
    monkeypatch.setattr(cli, "JevBackend", lambda **kwargs: backend)
    dataset, output = tmp_path / "samples.json", tmp_path / "report.json"
    rows = (
        []
        if outcome == "empty"
        else [
            {"input": "Q", "response": "A", "contexts": None if outcome == "skip" else ["Evidence"]}
        ]
    )
    dataset.write_text(json.dumps(rows))
    args = [
        "evaluate",
        str(dataset),
        "--metrics",
        "faithfulness",
        "--threshold",
        "0.8",
        "--missing",
        "skip",
        "--errors",
        "record",
        "--fail-on-failure",
        "--output",
        str(output),
    ]
    assert main(args) == expected_exit
    report = json.loads(output.read_text())
    if outcome == "empty":
        assert report["results"] == []
    else:
        row = report["results"][0]
        assert row["metrics"]["faithfulness"]["status"] == expected_status
        assert row["passed"] is ({"pass": True, "fail": False}.get(outcome))
    assert backend.sessions == backend.closed == int(outcome not in ("skip", "empty"))
    assert "private provider detail" not in output.read_text()
    assert "private provider detail" not in capsys.readouterr().out


@pytest.mark.parametrize("command", ["evaluate", "calibrate"])
def test_cli_rejects_unknown_backend(command, tmp_path):
    with pytest.raises(SystemExit) as caught:
        main([command, "rows.json", "--backend", "unknown", "--output", str(tmp_path / "x")])
    assert caught.value.code == 2


@pytest.mark.parametrize("command", ["evaluate", "calibrate"])
def test_cli_missing_openai_extra_gives_installation_instruction(command, monkeypatch, capsys):
    import typed_evals.cli as cli

    def unavailable(**kwargs):
        raise ImportError("private provider detail")

    monkeypatch.setattr(cli, "OpenAIDecisionsBackend", unavailable)
    assert main([command, "rows.json", "--backend", "openai-decisions", "--output", "x.json"]) == 2
    stderr = capsys.readouterr().err
    assert "python -m pip install 'typed-evals[openai]'" in stderr
    assert "private provider detail" not in stderr
