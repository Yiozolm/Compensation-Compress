from typing import Optional, Tuple
import torch
from torch import Tensor

from compressai.entropy_models import GaussianConditional

__all__ = [
    "GaussianConditionalBiased"
]


class GaussianConditionalBiased(GaussianConditional):
    def __init__(self, bias, **kwargs):
        super().__init__(**kwargs)
        self.bias = float(0.0)
        if not bias or not isinstance(bias, torch.Tensor):
            raise TypeError('bias must be a torch.Tensor')
        if bias:
            self.bias = float(bias.item())

        if self.bias < -0.5 or self.bias > 0.5:
            raise ValueError('bias must be between -0.5 and 0.5')

    def update(self):
        multiplier = -self._standardized_quantile(self.tail_mass / 2)
        pmf_center = torch.ceil(self.scale_table * multiplier).int()
        pmf_length = 2 * pmf_center + 1
        max_length = torch.max(pmf_length).item()

        device = pmf_center.device
        samples = torch.abs(
            torch.arange(max_length, device=device).int() - pmf_center[:, None]
        )
        samples_scale = self.scale_table.unsqueeze(1)
        samples = samples.float()
        samples_scale = samples_scale.float()
        upper = self._standardized_cumulative((0.5 + self.bias - samples) / samples_scale)
        lower = self._standardized_cumulative((-0.5 + self.bias - samples) / samples_scale)
        pmf = upper - lower

        tail_mass = 2 * lower[:, :1]

        quantized_cdf = torch.Tensor(len(pmf_length), max_length + 2)
        quantized_cdf = self._pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
        self._quantized_cdf = quantized_cdf
        self._offset = -pmf_center
        self._cdf_length = pmf_length + 2

    def _likelihood(
            self, inputs: Tensor, scales: Tensor, means: Optional[Tensor] = None
    ) -> Tensor:
        half = float(0.5)

        if means is not None:
            values = inputs - means
        else:
            values = inputs

        scales = self.lower_bound_scale(scales)  # σ下界，防止梯度消失

        # 使用erfc函数计算出高斯模型中潜在表示y中每个点的概率
        values = torch.abs(values)
        upper = self._standardized_cumulative((half - values + self.bias) / scales)
        lower = self._standardized_cumulative((-half - values + self.bias) / scales)
        likelihood = upper - lower

        return likelihood

    def forward(
            self,
            inputs: Tensor,
            scales: Tensor,
            means: Optional[Tensor] = None,
            bias: Optional[float] = None,
            training: Optional[bool] = None,
    ) -> Tuple[Tensor, Tensor]:
        if training is None:
            training = self.training
        if bias and 0.5 <= bias <= 0.5:
            self.bias = bias
        outputs = self.quantize(inputs + self.bias, "noise" if training else "dequantize", means)  # outputs返回量化参数
        likelihood = self._likelihood(outputs, scales, means)  # 每个待编码值出现的概率估计
        if self.use_likelihood_bound:
            likelihood = self.likelihood_lower_bound(likelihood)
        return outputs, likelihood  # 量化后的值outputs、每个待编码值的出现概率的估计likelihood

    def compress(self, inputs, indexes, means=None):
        """
        Compress input tensors to char strings.

        Args:
            inputs (torch.Tensor): input tensors
            indexes (torch.IntTensor): tensors CDF indexes
            means (torch.Tensor, optional): optional tensor means
        """
        symbols = self.quantize(inputs + self.bias, "symbols", means)

        if len(inputs.size()) < 2:
            raise ValueError(
                "Invalid `inputs` size. Expected a tensor with at least 2 dimensions."
            )

        if inputs.size() != indexes.size():
            raise ValueError("`inputs` and `indexes` should have the same size.")

        self._check_cdf_size()
        self._check_cdf_length()
        self._check_offsets_size()

        strings = []
        for i in range(symbols.size(0)):
            rv = self.entropy_coder.encode_with_indexes(
                symbols[i].reshape(-1).int().tolist(),
                indexes[i].reshape(-1).int().tolist(),
                self._quantized_cdf.tolist(),
                self._cdf_length.reshape(-1).int().tolist(),
                self._offset.reshape(-1).int().tolist(),
            )
            strings.append(rv)
        return strings