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


def test_core_exports_and_jev_work_without_openai(tmp_path):
    script = """
import asyncio
import importlib.abc
import sys
from contextlib import asynccontextmanager

class NoOpenAI(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'openai':
            raise ImportError('OpenAI deliberately unavailable')

sys.meta_path.insert(0, NoOpenAI())
from typed_evals import (
    EvaluationSample, Evaluator, JevBackend, JudgeResponse, OpenAIDecisionsBackend,
)
from typed_evals.backends import OpenAIDecisionsBackend as BackendExport
import httpx2
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

assert BackendExport is OpenAIDecisionsBackend
assert 'openai' not in sys.modules

class CustomBackend:
    model = 'custom'
    @asynccontextmanager
    async def session(self):
        yield self
    async def judge(self, state, questions):
        return JudgeResponse(model=self.model, answers={
            name: {'type': 'noul', 'noul': 0.9} for name in questions
        })

async def run():
    sample = EvaluationSample(input='Question', response='Answer')
    assert (await Evaluator(backend=CustomBackend()).aevaluate_one(sample)).passed
    def handler(request):
        return httpx2.Response(200, json={
            'model': 'jev-1.13.0', 'usage': {'input_tokens': 2, 'output_tokens': 0},
            'answers': {'answer_relevancy': {'type': 'noul', 'noul': 0.9}},
        })
    async with AsyncTypeSafeClient(
        api_key='offline-test', transport=httpx2.MockTransport(handler),
        retry=RetryPolicy(max_retries=0),
    ) as client:
        assert (await Evaluator(backend=JevBackend(client=client)).aevaluate_one(sample)).passed
    try:
        async with OpenAIDecisionsBackend().session():
            pass
    except ImportError as exc:
        assert 'openai' in str(exc).lower()
        assert 'install' in str(exc).lower()
    else:
        raise AssertionError('Missing OpenAI installation was not reported')

asyncio.run(run())
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
