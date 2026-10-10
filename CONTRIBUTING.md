# Contributing

Report ordinary bugs and propose changes through
[GitHub issues](https://github.com/TrustifAI/typed_evals/issues). Include the package
version, a minimal reproduction, expected behavior, and actual behavior. Use
synthetic evidence and remove credentials. For vulnerabilities, follow the
[security policy](SECURITY.md).

## Development environment

Use Python 3.11 or later. From a repository checkout, create and activate a virtual
environment, then install the dependencies used by the main CI job:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[openai,calibration,dev,langchain,agent-framework]'
```

CrewAI currently requires OpenAI SDK `<3`, while native Decisions requires
`>=3.26.0,<4`. Test CrewAI in a separate environment with
`python -m pip install -e '.[calibration,dev,crewai]'`. See the
[CI workflow](.github/workflows/test.yml) for the Python matrix and adapter jobs.

## Checks before a pull request

```bash
python -m pytest -m "not live" --cov=typed_evals --cov-report=term-missing
python -m ruff check .
python -m ruff format --check .
python -m mypy
python -m build
```

CI runs tests, formatting, lint, type checks, and the package build.
The normal suite uses fake judges and mocked provider HTTP and needs no API keys
or paid requests. Tests for optional frameworks are skipped when their extras are
absent. In the separate CrewAI environment, run its CI subset:

```bash
python -m mypy --follow-imports=silent typed_evals/adapters/crewai.py
python -m pytest -m "not live" tests/test_crewai_adapter.py tests/test_tool_guard.py tests/test_runtime.py tests/test_evaluator.py tests/test_cli_backends.py
```

Keep changes focused. Add meaningful tests for changed behavior, update examples
and reference docs when public contracts change, and describe how you validated
the change in the pull request. Judge accuracy claims need representative labeled
data; mocked contract tests establish implementation behavior.

## Optional live smoke tests

Live tests make billable provider requests. Set the corresponding API key and
opt-in flag explicitly:

```bash
TYPED_EVALS_LIVE=1 python -m pytest tests/test_live.py::test_live_jev_smoke -m live
TYPED_EVALS_OPENAI_LIVE=1 python -m pytest tests/test_live.py::test_live_openai_decisions_smoke -m live
```

Jev uses `TYPESAFE_API_KEY`; Decisions uses `OPENAI_API_KEY`. Model overrides are
`TYPED_EVALS_MODEL` and `TYPED_EVALS_OPENAI_MODEL`, respectively. Live smoke tests
check the service contract; they do not establish metric accuracy.

## Releases

Maintainers should update the [changelog](CHANGELOG.md) and follow the
[publishing guide](docs/PUBLISHING.md). Release tags must match the package version;
the publishing workflow validates, builds, and publishes through PyPI Trusted Publishing.
Published versions are listed on [PyPI](https://pypi.org/project/typed-evals/#history).
