import json
from types import SimpleNamespace

import pytest

from typed_evals.backends import JevBackend, OpenAIDecisionsBackend
from typed_evals.cli import main


@pytest.mark.parametrize("command", ["evaluate", "calibrate"])
@pytest.mark.parametrize(
    "backend_name,backend_class,default_model",
    [
        ("jev", JevBackend, "jev-1.13.0"),
        ("openai-decisions", OpenAIDecisionsBackend, "gpt-6-luna"),
    ],
)
@pytest.mark.parametrize("model", [None, "explicit-model"])
def test_cli_selects_backend_and_its_default_model(
    command, backend_name, backend_class, default_model, model, tmp_path, monkeypatch
):
    import typed_evals.cli as cli

    captured = {}

    class Pipeline:
        def __init__(self, metrics, **kwargs):
            captured.update(kwargs)

        def fit(self, rows, validation_data=None):
            captured["rows"] = rows
            return SimpleNamespace(model_dump_json=lambda **kwargs: "{}")

        def save_calibration(self, path):
            output.write_text("{}")

        def evaluate(self, rows):
            captured["rows"] = rows
            return SimpleNamespace(
                summary={},
                results=[SimpleNamespace(passed=True)],
                save=lambda path: output.write_text("{}"),
            )

    # Backend construction needs no API key or connection. Keep the real constructors
    # so this verifies their defaults, rather than duplicating those defaults in a fake.
    other_class = OpenAIDecisionsBackend if backend_name == "jev" else JevBackend
    monkeypatch.setattr(cli, backend_class.__name__, backend_class)

    def unexpected_backend(**kwargs):
        raise AssertionError("CLI selected the wrong backend")

    monkeypatch.setattr(cli, other_class.__name__, unexpected_backend)
    monkeypatch.setattr(cli, "EvaluationPipeline", Pipeline)
    sample = {"input": "Question", "response": "Answer"}
    payload = [
        {"sample": sample, "labels": {"answer_relevancy": 1}} if command == "calibrate" else sample
    ]
    dataset = tmp_path / "rows.json"
    dataset.write_text(json.dumps(payload))
    output = tmp_path / "output.json"
    args = [command, str(dataset), "--backend", backend_name, "--output", str(output)]
    if model is not None:
        args.extend(["--model", model])
    if command == "evaluate":
        args.extend(["--missing", "skip", "--errors", "record", "--fail-on-failure"])
    assert main(args) == 0
    assert isinstance(captured["backend"], backend_class)
    assert captured["backend"].model == (model or default_model)
    assert output.exists()
    assert len(captured["rows"]) == 1
    if command == "evaluate":
        assert captured["missing"] == "skip"
        assert captured["errors"] == "record"


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
