"""System One CLI configuration reaches either pipeline without implicit credentials."""

import json
from types import SimpleNamespace

import pytest

from typed_evals.cli import main


@pytest.fixture
def cli_pipeline(tmp_path, monkeypatch):
    import typed_evals.cli as cli

    captured = {}

    def backend(**kwargs):
        captured["backend_kwargs"] = kwargs
        return SimpleNamespace(**kwargs)

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

    output = tmp_path / "output.json"
    monkeypatch.setattr(cli, "SystemOneBackend", backend)
    monkeypatch.setattr(cli, "EvaluationPipeline", Pipeline)
    return captured, output


def arguments(command, tmp_path, output):
    sample = {"input": "Question", "response": "Answer"}
    rows = [
        {"sample": sample, "labels": {"answer_relevancy": 1}} if command == "calibrate" else sample
    ]
    dataset = tmp_path / "rows.json"
    dataset.write_text(json.dumps(rows))
    return [
        command,
        str(dataset),
        "--backend",
        "systemone",
        "--model",
        "community/decision-model",
        "--base-url",
        "https://decision.example.test/",
        "--output",
        str(output),
    ]


@pytest.mark.parametrize("command", ["evaluate", "calibrate"])
def test_cli_systemone_without_auth_ignores_ambient_provider_keys(
    command, cli_pipeline, tmp_path, monkeypatch
):
    captured, output = cli_pipeline
    monkeypatch.setenv("TYPESAFE_API_KEY", "ambient-typesafe-secret")
    monkeypatch.setenv("HF_TOKEN", "ambient-huggingface-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-openai-secret")
    assert main(arguments(command, tmp_path, output)) == 0
    assert captured["backend_kwargs"] == {
        "model": "community/decision-model",
        "base_url": "https://decision.example.test/",
        "api_key": None,
    }
    assert len(captured["rows"]) == 1
    assert output.exists()


@pytest.mark.parametrize("command", ["evaluate", "calibrate"])
def test_cli_systemone_reads_only_named_api_key(
    command, cli_pipeline, tmp_path, monkeypatch, capsys
):
    captured, output = cli_pipeline
    monkeypatch.setenv("MY_JUDGE_TOKEN", "named-server-secret")
    monkeypatch.setenv("HF_TOKEN", "ambient-huggingface-secret")
    args = arguments(command, tmp_path, output) + ["--api-key-env", "MY_JUDGE_TOKEN"]
    assert main(args) == 0
    assert captured["backend_kwargs"]["api_key"] == "named-server-secret"
    printed = capsys.readouterr()
    assert "named-server-secret" not in printed.out + printed.err


@pytest.mark.parametrize("command", ["evaluate", "calibrate"])
@pytest.mark.parametrize("option", ["--model", "--base-url"])
@pytest.mark.parametrize("value", [None, "", "  "])
def test_cli_systemone_requires_nonempty_model_and_base_url(
    command, option, value, cli_pipeline, tmp_path, capsys
):
    captured, output = cli_pipeline
    args = arguments(command, tmp_path, output)
    index = args.index(option)
    if value is None:
        del args[index : index + 2]
    else:
        args[index + 1] = value
    with pytest.raises(SystemExit) as caught:
        main(args)
    assert caught.value.code == 2
    assert f"requires {option}" in capsys.readouterr().err
    assert not captured
    assert not output.exists()


@pytest.mark.parametrize("command", ["evaluate", "calibrate"])
@pytest.mark.parametrize("value", [None, "", "  "])
def test_cli_systemone_rejects_missing_or_empty_named_api_key_before_pipeline(
    command, value, cli_pipeline, tmp_path, monkeypatch, capsys
):
    captured, output = cli_pipeline
    if value is None:
        monkeypatch.delenv("MY_JUDGE_TOKEN", raising=False)
    else:
        monkeypatch.setenv("MY_JUDGE_TOKEN", value)
    args = arguments(command, tmp_path, output) + ["--api-key-env", "MY_JUDGE_TOKEN"]
    with pytest.raises(SystemExit) as caught:
        main(args)
    assert caught.value.code == 2
    assert "--api-key-env must name a nonempty environment variable" in capsys.readouterr().err
    assert not captured
    assert not output.exists()


@pytest.mark.parametrize("command", ["evaluate", "calibrate"])
@pytest.mark.parametrize("backend", ["jev", "openai-decisions"])
@pytest.mark.parametrize(
    "option,value",
    [("--base-url", "https://decision.example.test/"), ("--api-key-env", "MY_JUDGE_TOKEN")],
)
def test_cli_rejects_systemone_options_for_other_backends(command, backend, option, value, capsys):
    with pytest.raises(SystemExit) as caught:
        main([command, "rows.json", "--backend", backend, "--output", "out.json", option, value])
    assert caught.value.code == 2
    assert "require --backend systemone" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["evaluate", "calibrate"])
def test_cli_systemone_provider_failure_redacts_api_key(
    command, cli_pipeline, tmp_path, monkeypatch, capsys
):
    import typed_evals.cli as cli

    _, output = cli_pipeline
    monkeypatch.setenv("MY_JUDGE_TOKEN", "private-server-secret")

    def fail(**kwargs):
        raise RuntimeError("provider echoed private-server-secret and private evidence")

    monkeypatch.setattr(cli, "SystemOneBackend", fail)
    args = arguments(command, tmp_path, output) + ["--api-key-env", "MY_JUDGE_TOKEN"]
    assert main(args) == 2
    printed = capsys.readouterr()
    assert "RuntimeError" in printed.err
    assert "private-server-secret" not in printed.out + printed.err
    assert "private evidence" not in printed.out + printed.err
    assert not output.exists()
