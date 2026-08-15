"""E-15.13.  Likelihood displacement, on the example E-15.6 constructs.

The blueprint asked for a three-completion TABULAR example.  There is none:
with one free logit per completion the DPO step adds +c to the chosen logit, so
its probability rises at every step.  Displacement needs the chosen and
rejected completions to SHARE parameters, which is exactly what two answers to
the same prompt do, so the example below gives each completion a feature vector.

The number worth noticing is not that both probabilities reach zero.  It is that
after forty steps the loss has moved about one per cent and the chosen
completion has lost most of its mass.
"""
import numpy as np
import pytest

from exercises.ch15.solution import dpo_step

# y_w and y_l are nearly identical; y_o is unlabelled and far along phi_w - phi_l
PHI = np.array([[1.0, 0.1], [1.0, -0.1], [0.0, 5.0]])
BETA, LR, STEPS = 0.1, 2.0, 3000


def _run():
    theta = np.zeros(2)
    z = PHI @ theta
    pref = np.exp(z - z.max())
    pref = pref / pref.sum()
    losses, pis = [], []
    for _ in range(STEPS):
        theta, loss, pi = dpo_step(theta, PHI, pref, BETA, LR)
        losses.append(loss)
        pis.append(pi)
    return np.array(losses), np.array(pis)


def test_chosen_logprob_falls():
    losses, pis = _run()
    assert np.all(np.diff(losses) < 0), "the loss must fall monotonically"
    assert losses[0] == pytest.approx(np.log(2.0), abs=1e-9)
    assert np.all(np.diff(pis[:, 0]) < 0), "pi(y_w) must fall at every step"
    assert np.all(np.diff(pis[:, 1]) < 0), "pi(y_l) must fall at every step"
    assert pis[-1, 0] < 1e-6 and pis[-1, 1] < 1e-6
    assert pis[-1, 2] > 0.999, "the unlabelled completion takes the mass"


def test_the_loss_barely_moves_while_the_policy_collapses():
    losses, pis = _run()
    assert (losses[0] - losses[40]) / losses[0] < 0.02
    assert pis[40, 0] < 0.10 * pis[0, 0]


def test_the_implicit_reward_still_ranks_the_pair_correctly():
    """Held-out preference accuracy is perfect throughout, which is why this
    failure is invisible to the obvious metric."""
    _, pis = _run()
    z = PHI @ np.zeros(2)
    pref = np.exp(z - z.max())
    pref = pref / pref.sum()
    rhat = BETA * np.log(pis[-1] / pref)
    assert rhat[0] > rhat[1]
