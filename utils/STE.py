import torch
import math

__all__ = ["LearnableBiasModule",
           "quantize_ste",
           "quantize_ste_biased",
           "round_with_STE"]


class LearnableBiasModule(torch.nn.Module):
    def __init__(self, init_val=0.0, init_type='tanh'):
        super().__init__()
        self.bias_param = torch.nn.Parameter(torch.tensor([init_val], dtype=torch.float32))
        self.type = init_type

    def forward(self):
        if self.type == 'tanh':
            return torch.tanh(self.bias_param) / 2
        elif self.type == 'erf':
            return ErfClampedBias.apply(self.bias_param)
        elif self.type == 'erfc':
            return (torch.erfc(self.bias_param)-1) / 2
        else:
            raise NotImplementedError


class ErfClampedBias(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bias_param: torch.Tensor):
        ctx.save_for_backward(bias_param)
        return 0.5 * torch.erf(bias_param)  # output in (-0.5, 0.5)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        bias_param, = ctx.saved_tensors
        grad_bias_param = grad_output * (1.0 / math.sqrt(math.pi)) * torch.exp(-bias_param ** 2)
        return grad_bias_param


def quantize_ste(x: torch.Tensor) -> torch.Tensor:
    """
    Rounding with non-zero gradients. Gradients are approximated by replacing
    the derivative by the identity function.

    Used in `"Lossy Image Compression with Compressive Autoencoders"
    <https://arxiv.org/abs/1703.00395>`_

    .. note::

        Implemented with the pytorch `detach()` reparametrization trick:

        `x_round = x_round - x.detach() + x`
    """
    return (torch.round(x) - x).detach() + x


class RoundWithSTE(torch.autograd.Function):
    """
    Special rounding that uses a straight-through estimator (STE) for backpropagation.
    The unofficial pytorch version code in `https://github.com/mandt-lab/improving-inference-for-neural-image-compression/blob/main/utils.py`

    Used in `"Improving Inference for Neural Image Compression"
    <https://arxiv.org/abs/2006.04240>`_
    """

    @staticmethod
    def forward(ctx, x, STE=None):
        """
        Forward pass: standard rounding operation.
        :param x: input tensor
        :param STE: type of proxy function whose gradient is used in place of round in the backward pass
        :return: rounded tensor
        """
        ctx.STE = STE  # Save STE type for backward pass
        return torch.round(x)  # Standard rounding operation

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: applies different STE variants to approximate gradients.
        :param grad_output: gradient from next layer
        :return: modified gradient (STE applied), None (for STE parameter)
        """
        STE = ctx.STE  # Retrieve stored STE type

        if STE is None or STE == 'identity':
            grad_input = grad_output  # Identity STE: Pass gradient unchanged
        elif STE == 'relu':
            grad_input = torch.relu(grad_output)  # ReLU STE: max{grad_output, 0}
        elif STE == 'crelu' or STE == 'clipped_relu':
            grad_input = torch.clamp(torch.relu(grad_output), 0., 1.)  # Clipped ReLU: min{max{grad_output, 0}, 1}
        else:
            raise NotImplementedError(f"STE type '{STE}' is not implemented.")

        return grad_input, None  # None for STE parameter as it's not differentiable


# Function wrapper for convenience
def round_with_STE(x, STE=None):
    """
    A wrapper function to apply the custom rounding operation with STE.
    :param x: input tensor
    :param STE: type of STE for backpropagation
    :return: rounded tensor with STE applied in backward pass
    """
    return RoundWithSTE.apply(x, STE)


class QuantizeSteBiased(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, bias=torch.Tensor) -> torch.Tensor:
        """
        Existing quantization round methods deploy quantization by rounding with STE
        round(x) = ceil(x) if x >= floor(x) + 0.5 else floor(x)

        We implement quantization with a biased round
        round(x) = ceil(x) if x >= floor(x) + 0.5 + bias else floor(x)

        Forward pass: modified rounding operation. we want to minimize quantization error
        :param x: input tensor
        :param bias: a trained parameter in network, range from -0.5 to 0.5
        :return: rounded tensor
        """
        ctx.save_for_backward(x, bias)
        return torch.round(x + bias)

    @staticmethod
    def backward(ctx, grad_output):
        # x, bias = ctx.saved_tensors
        grad_x = grad_output
        grad_bias = grad_output.sum(dim=tuple(range(1, grad_output.ndim)), keepdim=True)  # optional sum
        return grad_x, grad_bias


# Function wrapper for convenience
def quantize_ste_biased(x, bias=0.0):
    """
    A wrapper function to apply the custom rounding operation with STE.
    :param x: input tensor
    :param bias:  a trained parameter in network, range from -0.5 to 0.5
    :return: rounded tensor with STE applied in backward pass
    """
    return QuantizeSteBiased.apply(x, bias)


# Example usage
if __name__ == "__main__":
    # x = torch.tensor([1.2, 2.8, 3.5, -1.7, -2.3], requires_grad=True)
    x = torch.normal(0.5, 1, size=(1, 255)) + torch.normal(0, 0.95, size=(1, 255))

    mean = torch.mean(x)

    y = round_with_STE(x, STE='clipped_relu')  # Try different STE modes: 'identity', 'relu', 'clipped_relu'
    # y.backward(torch.ones_like(x))  # Compute gradients

    z = quantize_ste(x - mean) + mean

    print("Input:", x.detach().numpy())
    print("Rounded Output:", y.detach().numpy())
    # print("Gradient:", x.grad.numpy())  # Display the computed gradients
    print('-----')
    print("Mean:", mean.detach().numpy())
    print("z:", z.detach().numpy())
    print('-----')

    biases = [-0.5 + 0.01 * n for n in range(0, 100)]

    min_mse = torch.tensor(1000000.0)
    index = -1
    for bias in biases:
        quantize_x = quantize_ste_biased(x, bias)
        mse = torch.sum((quantize_x - x) ** 2)
        if mse < min_mse:
            min_mse = min(min_mse, mse)
            index = bias
        print("--Biased:", bias,"--")
        print("quantized_x:",quantize_x.detach().numpy())
        print("MSE:", mse)
        print("--")
    print(min_mse)
    print(index)
