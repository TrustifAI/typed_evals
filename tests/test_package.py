"""Exercise the installed distribution and both CLI entry points away from the repo."""

import os
import subprocess
import sys
import sysconfig
from importlib.metadata import distribution
from importlib.resources import files
from pathlib import Path

import pytest

import typed_evals
from typed_evals.cli import main


def test_installed_distribution_exposes_cli_and_typing_marker():
    installed = distribution("typed_evals")
    assert installed.version == typed_evals.__version__
    entry_points = {
        entry.name: entry for entry in installed.entry_points if entry.group == "console_scripts"
    }
    assert set(entry_points) == {"typed_evals"}
    assert entry_points["typed_evals"].load() is main
    assert files("typed_evals").joinpath("py.typed").is_file()


@pytest.mark.parametrize("entry_point", ["module", "console"])
def test_installed_cli_help_outside_repository(entry_point, tmp_path):
    if entry_point == "module":
        command = [sys.executable, "-m", "typed_evals"]
    else:
        executable = "typed_evals.exe" if os.name == "nt" else "typed_evals"
        command = [str(Path(sysconfig.get_path("scripts")) / executable)]
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [*command, "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "usage: typed_evals" in result.stdout
    assert "{evaluate,calibrate}" in result.stdout


def test_core_and_adapter_modules_do_not_require_framework_installs(tmp_path):
    script = """
import importlib.abc
import sys

class NoFrameworks(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'crewai', 'agent_framework', 'langchain', 'langgraph'}:
            raise ImportError('Framework deliberately unavailable')

sys.meta_path.insert(0, NoFrameworks())
from typed_evals import Evaluator, Metric
from typed_evals.evaluation import EvaluationPipeline
from typed_evals.metrics import Faithfulness
from typed_evals.runtime import RuntimeGuard
from typed_evals.adapters import agent_framework, crewai, langchain
assert isinstance(Faithfulness(), Metric)
try:
    crewai.guard_tool(policy='Owned only', input='Read', contexts=['Owned'])
except ImportError as exc:
    assert 'typed_evals[crewai]' in str(exc)
else:
    raise AssertionError('Missing CrewAI installation was not reported')
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
