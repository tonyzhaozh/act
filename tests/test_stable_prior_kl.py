import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'detr'))

from decimal import Decimal, localcontext

from policy import kl_divergence


def decimal_kl(mu, logvar):
    # High precision is independent of the implementation's cancellation fix.
    with localcontext() as context:
        context.prec = 70
        m = Decimal.from_float(float(mu))
        ell = Decimal.from_float(float(logvar))
        return float((m * m + ell.exp() - Decimal(1) - ell) / Decimal(2))


class StablePriorKLTest(unittest.TestCase):
    def test_near_prior_values_are_nonnegative_and_match_high_precision(self):
        for dtype in (torch.float32, torch.float64):
            logvar = torch.tensor([[1e-9, -1e-9, 1e-6, -1e-6, 1e-4, -1e-4, .003, -.003]], dtype=dtype)
            mu = torch.zeros_like(logvar)
            expected = torch.tensor([[decimal_kl(m, ell) for m, ell in zip(mu[0], logvar[0])]], dtype=dtype)
            total, dimension, mean = kl_divergence(mu, logvar)
            self.assertTrue((dimension > 0).all().item())
            torch.testing.assert_close(dimension, expected[0], rtol=2e-6, atol=0)
            torch.testing.assert_close(total, expected.sum(1), rtol=2e-6, atol=0)
            torch.testing.assert_close(mean, expected.mean(1), rtol=2e-6, atol=0)

    def test_first_and_second_derivatives_remain_correct_near_the_prior(self):
        logvar = torch.tensor([[1e-9, -1e-9, 1e-6, -.003]], dtype=torch.float32, requires_grad=True)
        mu = torch.zeros_like(logvar, requires_grad=True)
        total, _, _ = kl_divergence(mu, logvar)
        gradient = torch.autograd.grad(total.sum(), logvar, create_graph=True)[0]
        curvature = torch.autograd.grad(gradient.sum(), logvar)[0]
        expected_gradient = torch.tensor([[.5 * math.expm1(float(value)) for value in logvar[0]]], dtype=logvar.dtype)
        expected_curvature = torch.tensor([[.5 * math.exp(float(value)) for value in logvar[0]]], dtype=logvar.dtype)
        torch.testing.assert_close(gradient, expected_gradient, rtol=2e-6, atol=0)
        torch.testing.assert_close(curvature, expected_curvature, rtol=2e-6, atol=0)

    def test_batch_reductions_and_far_prior_values_preserve_the_analytic_objective(self):
        mu = torch.tensor([[.3, -.7, 2.], [-.8, .2, 1.]], dtype=torch.float64, requires_grad=True)
        logvar = torch.tensor([[-3., .1, 2.], [1., -2., .03]], dtype=torch.float64, requires_grad=True)
        expected = torch.tensor([[decimal_kl(m, ell) for m, ell in zip(row_m, row_l)] for row_m, row_l in zip(mu, logvar)], dtype=mu.dtype)
        total, dimension, mean = kl_divergence(mu, logvar)
        torch.testing.assert_close(total, expected.sum(1).mean().reshape(1), rtol=2e-14, atol=2e-14)
        torch.testing.assert_close(dimension, expected.mean(0), rtol=2e-14, atol=2e-14)
        torch.testing.assert_close(mean, expected.mean().reshape(1), rtol=2e-14, atol=2e-14)
        grad_mu, grad_logvar = torch.autograd.grad(total.sum(), (mu, logvar))
        torch.testing.assert_close(grad_mu, mu / mu.shape[0], rtol=2e-14, atol=2e-14)
        torch.testing.assert_close(grad_logvar, .5 * logvar.expm1() / mu.shape[0], rtol=2e-14, atol=2e-14)

    def test_extreme_negative_log_variance_and_exact_prior_are_finite(self):
        logvar = torch.tensor([[-1000., 0.]], requires_grad=True)
        mu = torch.zeros_like(logvar)
        total, dimension, _ = kl_divergence(mu, logvar)
        torch.testing.assert_close(dimension, torch.tensor([499.5, 0.]), rtol=0, atol=0)
        gradient = torch.autograd.grad(total.sum(), logvar)[0]
        torch.testing.assert_close(gradient, torch.tensor([[-.5, 0.]]), rtol=0, atol=0)


import math

if __name__ == '__main__':
    unittest.main()
