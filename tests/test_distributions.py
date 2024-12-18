import unittest
import torch
import math
from torch.distributions import Normal
from torch.testing import assert_close
from dreamer.distributions import (
    SampleDist,
    OneHotDist,
    TruncatedNormalDist,
    TanhTransform,
)


class TestSampleDist(unittest.TestCase):
    def test_mean(self):
        dist = Normal(loc=0.0, scale=1.0)
        sampler = SampleDist(dist)
        # The mean of a Normal(0,1) is 0. We'll allow a small margin of error.
        est_mean = sampler.mean()
        self.assertAlmostEqual(est_mean.item(), 0.0, places=0)

    def test_mode(self):
        # For a Normal distribution, mode = mean.
        dist = Normal(loc=2.0, scale=0.5)
        sampler = SampleDist(dist)
        mode_est = sampler.mode().item()
        # Mode should be close to 2.0
        self.assertAlmostEqual(mode_est, 2.0, places=1)

    def test_entropy(self):
        dist = Normal(loc=0.0, scale=1.0)
        sampler = SampleDist(dist)
        # Entropy of N(0,1) is 0.5*log(2*pi*e) ~ 1.4189
        true_entropy = 0.5 * math.log(2 * math.pi * math.e)
        est_entropy = sampler.entropy().item()
        self.assertAlmostEqual(est_entropy, true_entropy, places=0)


class TestOneHotDist(unittest.TestCase):
    def test_mode(self):
        logits = torch.tensor([1.0, 2.0, 0.5])
        dist = OneHotDist(logits=logits)
        # The category with highest logit is index 1
        mode = dist.mode()
        # mode should be one-hot at index 1
        self.assertTrue(torch.allclose(mode, torch.tensor([0, 1, 0], dtype=mode.dtype)))

    def test_sample_shape(self):
        logits = torch.tensor([1.0, 1.0, 1.0])
        dist = OneHotDist(logits=logits)
        samples = dist.sample((5,))
        self.assertEqual(samples.shape, (5, 3))

    def test_straight_through_grad(self):
        # Check differentiability
        logits = torch.tensor([0.5, 1.0, -1.0], requires_grad=True)
        dist = OneHotDist(logits=logits)
        sample = dist.sample()
        weights = torch.tensor([1.0, 2.0, 3.0])
        # sample should be differentiable w.r.t logits
        loss = (sample * weights).sum()
        loss.backward()
        # Gradient should not be zero in general
        self.assertIsNotNone(logits.grad)
        self.assertFalse(torch.allclose(logits.grad, torch.zeros_like(logits.grad)))
        self.assertTrue((logits.grad.abs() > 1e-9).any(), f"Gradients are zero!")


class TestTruncatedNormalDist(unittest.TestCase):
    def setUp(self):
        self.loc = 0.0
        self.scale = 1.0
        self.low = -1.0
        self.high = 1.0
        self.dist = TruncatedNormalDist(self.loc, self.scale, self.low, self.high)

    def test_sample_range(self):
        samples = self.dist.sample((1000,))
        self.assertTrue((samples >= self.low).all().item())
        self.assertTrue((samples <= self.high).all().item())

    def test_log_prob_outside_bounds(self):
        val = torch.tensor(2.0)
        lp = self.dist.log_prob(val)
        self.assertTrue(torch.isinf(lp))
        self.assertTrue(lp < 0.0)  # negative infinity

    def test_log_prob_inside_bounds(self):
        val = torch.tensor(0.0)
        lp = self.dist.log_prob(val)
        self.assertFalse(torch.isinf(lp))

    def test_clipping(self):
        # With clipping, values at the edge should be adjusted straight-through
        d = TruncatedNormalDist(0.0, 1.0, -1.0, 1.0, clip=0.01)
        s = d.sample((10,))
        # Check that the samples are never exactly outside [-0.99, 0.99]
        self.assertTrue((s >= -0.99).all().item())
        self.assertTrue((s <= 0.99).all().item())


class TestTanhTransform(unittest.TestCase):
    def test_forward_inverse(self):
        transform = TanhTransform()
        x = torch.randn(10)
        y = transform(x)
        x_inv = transform.inv(y)
        assert_close(x, x_inv, atol=1e-5, rtol=1e-5)

    def test_log_abs_det_jacobian(self):
        transform = TanhTransform()
        x = torch.randn(10)
        y = transform(x)
        ladj = transform.log_abs_det_jacobian(x, y)
        # Just verify it returns a finite tensor
        self.assertTrue(torch.isfinite(ladj).all())

    def test_bijectivity(self):
        transform = TanhTransform()
        # The transform should map R to (-1,1)
        x = torch.tensor([-10.0, 0.0, 10.0])
        y = transform(x)
        self.assertTrue((y >= -1.0).all() and (y <= 1.0).all())


if __name__ == "__main__":
    unittest.main()
