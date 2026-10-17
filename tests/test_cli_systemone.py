"""System One CLI configuration reaches either pipeline without implicit credentials."""

import json
from types import SimpleNamespace

import httpx2
import pytest
from conftest import calibrated_rows

from typed_evals.backends import systemone
from typed_evals.cli import main


@pytest.fixture
def offline_systemone(tmp_path, monkeypatch):
    captured = SimpleNamespace(requests=[], clients=[], fail=False, output=tmp_path / "output.json")

    def handler(request):
        captured.requests.append(request)
        if captured.fail:
            return httpx2.Response(
                403, text="provider echoed private-server-secret and private evidence"
            )
        body = json.loads(request.content)
        return httpx2.Response(
            200,
            json={
                "model": body["model"],
                "answers": {
                    name: {"type": "noul", "noul": float(body["state"]["response"])}
                    for name in body["questions"]
                },
            },
        )

    real_client = httpx2.AsyncClient

    def offline_client(**kwargs):
        client = real_client(transport=httpx2.MockTransport(handler), **kwargs)
        captured.clients.append(client)
        return client

    monkeypatch.setattr(systemone.httpx2, "AsyncClient", offline_client)
    return captured


def arguments(command, tmp_path, output):
    rows = (
        [row.model_dump(mode="json") for row in calibrated_rows()]
        if command == "calibrate"
        else [{"input": "Question", "response": "0.8"}]
    )
    dataset = tmp_path / "rows.json"
    dataset.write_text(json.dumps(rows))
    args = [
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
    return args


def assert_saved_result(command, captured):
    artifact = json.loads(captured.output.read_text())
    if command == "evaluate":
        row = artifact["results"][0]
        assert row["model"] == "community/decision-model"
        assert row["passed"] is True
        assert row["metrics"]["answer_relevancy"]["raw_score"] == 0.8
        assert len(captured.requests) == 1
    else:
        assert (
            artifact["requested_model"] == artifact["observed_model"] == "community/decision-model"
        )
        assert artifact["curves"]["answer_relevancy"]["n_samples"] == 40
        assert artifact["report"]["metrics"]["answer_relevancy"]["validation_samples"] == 40
        assert len(captured.requests) == 80
    assert captured.clients
    assert all(client.is_closed for client in captured.clients)
    for request in captured.requests:
        assert request.method == "POST"
        assert str(request.url) == "https://decision.example.test/v1/systemone"
        assert json.loads(request.content)["model"] == "community/decision-model"


@pytest.mark.parametrize("command", ["evaluate", "calibrate"])
def test_cli_systemone_without_auth_ignores_ambient_provider_keys(
    command, offline_systemone, tmp_path, monkeypatch
):
    captured = offline_systemone
    monkeypatch.setenv("TYPESAFE_API_KEY", "ambient-typesafe-secret")
    monkeypatch.setenv("HF_TOKEN", "ambient-huggingface-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-openai-secret")
    assert main(arguments(command, tmp_path, captured.output)) == 0
    assert_saved_result(command, captured)
    assert all("authorization" not in request.headers for request in captured.requests)


@pytest.mark.parametrize("command", ["evaluate", "calibrate"])
def test_cli_systemone_reads_only_named_api_key(
    command, offline_systemone, tmp_path, monkeypatch, capsys
):
    captured = offline_systemone
    monkeypatch.setenv("MY_JUDGE_TOKEN", "named-server-secret")
    monkeypatch.setenv("HF_TOKEN", "ambient-huggingface-secret")
    args = arguments(command, tmp_path, captured.output) + ["--api-key-env", "MY_JUDGE_TOKEN"]
    assert main(args) == 0
    assert_saved_result(command, captured)
    assert all(
        request.headers["authorization"] == "Bearer named-server-secret"
        for request in captured.requests
    )
    assert "named-server-secret" not in captured.output.read_text()
    printed = capsys.readouterr()
    assert "named-server-secret" not in printed.out + printed.err


@pytest.mark.parametrize("command", ["evaluate", "calibrate"])
@pytest.mark.parametrize("option", ["--model", "--base-url"])
@pytest.mark.parametrize("value", [None, "", "  "])
def test_cli_systemone_requires_nonempty_model_and_base_url(
    command, option, value, offline_systemone, tmp_path, capsys
):
    captured = offline_systemone
    args = arguments(command, tmp_path, captured.output)
    index = args.index(option)
    if value is None:
        del args[index : index + 2]
    else:
        args[index + 1] = value
    with pytest.raises(SystemExit) as caught:
        main(args)
    assert caught.value.code == 2
    assert f"requires {option}" in capsys.readouterr().err
    assert not captured.clients
    assert not captured.requests
    assert not captured.output.exists()


@pytest.mark.parametrize("command", ["evaluate", "calibrate"])
@pytest.mark.parametrize("value", [None, "", "  "])
def test_cli_systemone_rejects_missing_or_empty_named_api_key_before_pipeline(
    command, value, offline_systemone, tmp_path, monkeypatch, capsys
):
    captured = offline_systemone
    if value is None:
        monkeypatch.delenv("MY_JUDGE_TOKEN", raising=False)
    else:
        monkeypatch.setenv("MY_JUDGE_TOKEN", value)
    args = arguments(command, tmp_path, captured.output) + ["--api-key-env", "MY_JUDGE_TOKEN"]
    with pytest.raises(SystemExit) as caught:
        main(args)
    assert caught.value.code == 2
    assert "--api-key-env must name a nonempty environment variable" in capsys.readouterr().err
    assert not captured.clients
    assert not captured.requests
    assert not captured.output.exists()


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
    command, offline_systemone, tmp_path, monkeypatch, capsys
):
    captured = offline_systemone
    captured.fail = True
    monkeypatch.setenv("MY_JUDGE_TOKEN", "private-server-secret")
    args = arguments(command, tmp_path, captured.output) + ["--api-key-env", "MY_JUDGE_TOKEN"]
    assert main(args) == 2
    printed = capsys.readouterr()
    assert "SystemOneError" in printed.err
    assert "private-server-secret" not in printed.out + printed.err
    assert "private evidence" not in printed.out + printed.err
    assert captured.requests
    assert all(client.is_closed for client in captured.clients)
    assert not captured.output.exists()
