"""Tests for gsnn.optim.BayesOptAgent."""

import pytest

try:
    from ax import ParameterType  # noqa: F401
    from gsnn.optim.BayesOpt import BayesOptAgent
except ImportError:
    pytest.skip("BayesOpt requires a compatible ax install", allow_module_level=True)

import torch


class _MockEnv:
    def run(self, action):
        return torch.tensor([action.float().mean().item()])


@pytest.mark.slow
def test_bayes_opt_step():
    env = _MockEnv()
    agent = BayesOptAgent(env, n_actions=3, warmup=2, verbose=False, suppress_warnings=True)
    agent.step()
    assert len(agent.rewards) == 1


@pytest.mark.slow
def test_get_best_action():
    env = _MockEnv()
    agent = BayesOptAgent(env, n_actions=2, warmup=1, verbose=False, suppress_warnings=True)
    agent.step()
    best = agent.get_best_action()
    assert best is not None
