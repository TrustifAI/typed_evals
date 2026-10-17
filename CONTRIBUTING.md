# Contributing

Report ordinary bugs and propose changes through
[GitHub issues](https://github.com/TrustifAI/typed_evals/issues). Include the package
version, a minimal reproduction, expected behavior, and actual behavior. Use
synthetic evidence and remove credentials. For vulnerabilities, follow the
[security policy](SECURITY.md).

## Development environment

Use Python 3.12 or later. From a repository checkout, create and activate a virtual
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
absent. Context relevance regressions cover deterministic segmentation,
probability-to-boolean unit decisions, the aggregate fraction, source-span
diagnostics, unavailable units, calibration, and all three bundled wire protocols.
They use real provider SDK serialization where applicable with only HTTP mocked.
In the separate CrewAI environment, run its CI subset:

```bash
python -m mypy --follow-imports=silent typed_evals/adapters/crewai.py
python -m pytest -m "not live" tests/test_crewai_adapter.py tests/test_tool_guard.py tests/test_runtime.py tests/test_evaluator.py tests/test_cli_backends.py
```

Keep changes focused. Add meaningful tests for changed behavior, update examples
and reference docs when public contracts change, and describe how you validated
the change in the pull request. Keep the current release version and
[changelog](CHANGELOG.md) in sync when behavior or the version changes; see
[releases](#releases). Judge accuracy claims need representative
labeled data; mocked contract tests establish implementation behavior.

When changing metrics, distinguish event probabilities, normalized ordinal
scores, and relevance fractions in results and validation reports. Preserve
calibration fingerprints for the exact rubric and execution recipe. Aggregate
metrics must retain every source unit in the denominator, report unavailable
units without producing a partial score, and isolate unrelated evidence. Source
diagnostics should retain offsets and typed verdicts without persisting raw
evidence or generating rationales.

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

The same opt-in flags also enable the live ToolSafety corpus in
[`tests/test_tool_safety_adversarial.py`](tests/test_tool_safety_adversarial.py).
Those cases check expected authorization judgments on synthetic evidence without
executing the proposed tools.

The same flags enable the built-in semantic regression corpus in
[`tests/test_metric_semantics_live.py`](tests/test_metric_semantics_live.py):

```bash
TYPED_EVALS_LIVE=1 python -m pytest tests/test_metric_semantics_live.py -m live
TYPED_EVALS_OPENAI_LIVE=1 python -m pytest tests/test_metric_semantics_live.py -m live
```

These are developer-authored expected cases, not independent human validation.
Use representative human-reviewed holdouts and the metric validation API before
making accuracy claims or selecting production thresholds.

## Releases

Update `[project].version` in `pyproject.toml` and the [changelog](CHANGELOG.md),
run the checks above, and review the resulting wheel and source distribution.
After the release changes are committed, publish a tag named exactly `v` followed
by that version, such as `v0.2.2` for version `0.2.2`.

The [publishing workflow](.github/workflows/publish.yml) requires that exact tag
match. It runs the shared test workflow, builds and validates both distributions,
and smoke-tests the installed wheel. Publishing waits for both jobs and uses
PyPI Trusted Publishing through the repository's `pypi` environment. Configure the
PyPI trusted publisher for this repository, workflow, and environment before the
first release. A manual workflow run on a branch validates and builds without
publishing; a matching release tag enables publication.
Published versions are listed on [PyPI](https://pypi.org/project/typed-evals/#history).
