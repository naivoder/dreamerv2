import torch
import math
import torch.nn.functional as F


class SampleDist:
    def __init__(self, dist, num_samples=100):
        self.dist = dist
        self.num_samples = num_samples

    @property
    def samples(self):
        return self.dist.sample((self.num_samples,))

    def mean(self):
        samples = self.samples
        return samples.mean(dim=0)

    def mode(self):
        samples = self.samples
        log_prob = self.dist.log_prob(samples)
        return samples[torch.argmax(log_prob, dim=0)]

    def entropy(self):
        samples = self.samples
        log_prob = self.dist.log_prob(samples)
        return -log_prob.mean(dim=0)


class OneHotDist(torch.distributions.OneHotCategorical):
    def __init__(
        self, logits=None, probs=None, dtype=torch.float32, validate_args=None
    ):
        super().__init__(logits=logits, probs=probs, validate_args=validate_args)
        self.dtype = dtype

    def mode(self):
        return super().mode.to(self.dtype)

    def sample(self, shape=torch.Size()):
        """
        Straight-through biased gradient estimator.
        Add (probs - probs.detach()) to make it differentiable w.r.t. logits
        """
        sample = super().sample(shape).to(self.dtype)
        return sample + (self.probs - self.probs.detach())


class TruncatedNormalDist(torch.distributions.Distribution):
    arg_constraints = {
        "loc": torch.distributions.constraints.real,
        "scale": torch.distributions.constraints.positive,
    }
    support = torch.distributions.constraints.real

    def __init__(self, loc, scale, low, high, clip=1e-6, mult=1, validate_args=None):
        self.loc = torch.as_tensor(loc)
        self.scale = torch.as_tensor(scale)
        self.low = torch.as_tensor(low)
        self.high = torch.as_tensor(high)
        self.clip = clip
        self.mult = mult
        self.normal = torch.distributions.Normal(self.loc, self.scale)
        super().__init__(self.loc.shape, validate_args=validate_args)

        # Precompute normalization constant for truncated normal
        self.log_z = torch.log(
            self.normal.cdf((self.high - self.loc) / self.scale)
            - self.normal.cdf((self.low - self.loc) / self.scale)
        )

    def sample(self, shape=torch.Size()):
        """Sample from normal and clamp to [low, high]"""
        event = self.normal.sample(shape)

        if self.clip:
            clipped = event.clamp(self.low + self.clip, self.high - self.clip)
            event = event + (clipped - event).detach()

        if self.mult:
            event *= self.mult

        return event

    def log_prob(self, value):
        """
        log p(x) = log(normal.pdf(x)) - log(Z), where Z = CDF(high) - CDF(low)
        If value is outside [low, high], log_prob = -inf
        """
        outside = (value < self.low) | (value > self.high)
        if outside.any():
            lp = torch.full_like(value, -math.inf)
            lp[~outside] = self.normal.log_prob(value[~outside]) - self.log_z
            return lp
        else:
            return self.normal.log_prob(value) - self.log_z


class TanhTransform(torch.distributions.Transform):
    domain = torch.distributions.constraints.real
    codomain = torch.distributions.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = 1

    def __init__(self):
        super().__init__()

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return torch.tanh(x)

    def _inverse(self, y):
        # atanh(y) = 0.5 * (log(1+y) - log(1-y))
        y = torch.clamp(y, -0.99999997, 0.99999997)
        return 0.5 * (torch.log1p(y) - torch.log1p(-y))

    def log_abs_det_jacobian(self, x, y):
        # log|dy/dx| = 2 * (log(2) - x - softplus(-2x))
        log2 = math.log(2.0)
        return 2.0 * (log2 - x - F.softplus(-2.0 * x))
