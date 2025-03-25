import torch
from compressai.models.google import ScaleHyperprior
from utils.STE import LearnableBiasModule, quantize_ste_biased, quantize_ste
from utils.entropy_models import GaussianConditionalBiased


class Compensation(ScaleHyperprior):
    def __init__(self, N, M, *args, **kwargs):
        super().__init__(N=N, M=M, **kwargs)

    def forward(self, input):
        half = float(0.5)
        y = self.g_a(input)
        z = self.h_a(torch.abs(y))
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat)
        y_hat = quantize_ste(y)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        y_hat = y_hat + torch.empty_like(y_hat).normal_(mean=0, std=1/6).clamp_(-half, half)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }


class CompensationWithBias(ScaleHyperprior):
    """
    A direct erf bias initial version
    """
    def __init__(self, N, M, bias_type='tanh',*args, **kwargs):
        super().__init__(N=N, M=M, **kwargs)
        self.bias = LearnableBiasModule(init_val=0.0, init_type=bias_type)
        self.gaussian_conditional = GaussianConditionalBiased(bias=self.bias)

    def forward(self, input):
        half = float(0.5)
        y = self.g_a(input)
        z = self.h_a(torch.abs(y))
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat)
        y_hat = quantize_ste_biased(y, bias=self.bias)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, bias=self.bias)
        y_hat = y_hat + torch.empty_like(y_hat).normal_(mean=0, std=1/6).clamp_(-half, half)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }