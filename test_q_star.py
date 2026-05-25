import torch
import torch.nn.functional as F

from distil_trainer import compute_q_star_logps


def _random_logps(batch, seqlen, vocab, seed=0, scale=1.0):
    torch.manual_seed(seed)
    logits = scale * torch.randn(batch, seqlen, vocab)
    return F.log_softmax(logits, dim=-1)


def _kl(log_a, log_b):
    return (log_a.exp() * (log_a - log_b)).sum(-1)


def test_rho_one_recovers_teacher():
    log_p = _random_logps(2, 4, 64, seed=0)
    log_T = _random_logps(2, 4, 64, seed=1)

    log_q_star, lam, gamma, kl_T_p = compute_q_star_logps(log_p, log_T, rho=1.0, n_iter=30)

    assert torch.allclose(log_q_star, log_T.float(), atol=1e-4), \
        "q* must equal T when rho = 1"
    assert torch.allclose(lam, torch.ones_like(lam), atol=1e-4), \
        f"lambda must be ~1 at rho=1, got mean {lam.mean().item():.4f}"
    assert torch.allclose(gamma, kl_T_p, atol=1e-6)


def test_lambda_monotone_in_rho():
    log_p = _random_logps(2, 4, 128, seed=0)
    log_T = _random_logps(2, 4, 128, seed=1, scale=2.0)

    rhos = [0.0, 0.25, 0.5, 0.75, 1.0]
    mean_lams = []
    for rho in rhos:
        _, lam, _, _ = compute_q_star_logps(log_p, log_T, rho=rho, n_iter=30)
        mean_lams.append(lam.mean().item())

    for i in range(len(mean_lams) - 1):
        assert mean_lams[i] <= mean_lams[i + 1] + 1e-3, \
            f"lambda not monotone: {mean_lams}"


def test_kl_qstar_less_than_kl_T():
    log_p = _random_logps(2, 4, 64, seed=0)
    log_T = _random_logps(2, 4, 64, seed=1, scale=2.0)

    kl_T_p = _kl(log_T.float(), log_p.float())
    for rho in [0.0, 0.25, 0.5, 0.75, 1.0]:
        log_q_star, _, _, _ = compute_q_star_logps(log_p, log_T, rho=rho, n_iter=40)
        kl_q_p = _kl(log_q_star, log_p.float())
        # should hold per-token,  with a small numerical difference
        assert (kl_q_p <= kl_T_p + 1e-4).all(), \
            f"rho={rho}: KL(q*||p) exceeds KL(T||p)"
