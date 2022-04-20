from typing import Tuple, Union

import oneflow as flow
import torch

_shape_t = Union[int, Tuple[int], flow._oneflow_internal.Size]


class LayerNorm(torch.nn.Module):
    """Applies Layer Normalization over a mini-batch of inputs as described in
    the paper `Layer Normalization <https://arxiv.org/abs/1607.06450>`__
    .. math::
        y = \\frac{x - \\mathrm{E}[x]}{ \\sqrt{\\mathrm{Var}[x] + \\epsilon}} * \\gamma + \\beta
    The mean and standard-deviation are calculated separately over the last
    certain number dimensions which have to be of the shape specified by
    :attr:`normalized_shape`.
    :math:`\\gamma` and :math:`\\beta` are learnable affine transform parameters of
    :attr:`normalized_shape` if :attr:`elementwise_affine` is ``True``.
    The standard-deviation is calculated via the biased estimator.
    .. note::
        Unlike Batch Normalization and Instance Normalization, which applies
        scalar scale and bias for each entire channel/plane with the
        :attr:`affine` option, Layer Normalization applies per-element scale and
        bias with :attr:`elementwise_affine`.
    This layer uses statistics computed from input data in both training and
    evaluation modes.
    Args:
        normalized_shape (int or list or oneflow.Size): input shape from an expected input of size
            .. math::
                [* \\times \\text{normalized_shape}[0] \\times \\text{normalized_shape}[1] \\times \\ldots \\times \\text{normalized_shape}[-1]]
            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        elementwise_affine: a boolean value that when set to ``True``, this module
            has learnable per-element affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.
    Shape:
        - Input: :math:`(N, *)`
        - Output: :math:`(N, *)` (same shape as input)
    For example:
    .. code-block:: python
        >>> import numpy as np
        >>> import oneflow as flow

        >>> input_arr = np.array(
        ...     [
        ...         [
        ...             [[-0.16046895, -1.03667831], [-0.34974465, 0.26505867]],
        ...             [[-1.24111986, -0.53806001], [1.72426331, 0.43572459]],
        ...         ],
        ...         [
        ...             [[-0.77390957, -0.42610624], [0.16398858, -1.35760343]],
        ...             [[1.07541728, 0.11008703], [0.26361224, -0.48663723]],
        ...         ],
        ...     ],
        ...     dtype=np.float32,
        ... )
        >>> x = flow.Tensor(input_arr)
        >>> m = flow.nn.LayerNorm(2)
        >>> y = m(x).numpy()
        >>> y
        array([[[[ 0.99997395, -0.99997395],
                 [-0.999947  ,  0.999947  ]],
        <BLANKLINE>
                [[-0.99995965,  0.9999595 ],
                 [ 0.99998784, -0.99998784]]],
        <BLANKLINE>
        <BLANKLINE>
               [[[-0.9998348 ,  0.99983466],
                 [ 0.9999914 , -0.9999914 ]],
        <BLANKLINE>
                [[ 0.9999785 , -0.9999785 ],
                 [ 0.9999646 , -0.9999646 ]]]], dtype=float32)
    """

    __constants__ = ["normalized_shape", "eps", "elementwise_affine"]
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(
        self,
        normalized_shape: _shape_t,
        eps: float = 1e-05,
        elementwise_affine: bool = True,
    ) -> None:
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = torch.nn.Parameter(torch.Tensor(*self.normalized_shape))
            self.bias = torch.nn.Parameter(torch.Tensor(*self.normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.reset_parameters()
        self.begin_norm_axis = 1
        self.begin_params_axis = 1

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, x):
        assert len(x.shape) > len(self.normalized_shape), "Input tensor dim must greater than normalized dim!"
        self.begin_norm_axis = len(x.shape) - len(self.normalized_shape)
        self.begin_params_axis = len(x.shape) - len(self.normalized_shape)

        for i in range(0, len(self.normalized_shape)):
            if x.shape[i + self.begin_params_axis] != self.normalized_shape[i]:
                raise RuntimeError(
                    f"Given normalized_shape={self.normalized_shape}, expected input with shape [*, {str(self.normalized_shape)[1:-1]}], but got input of size {x.shape}"
                )

        if not x.is_cuda:
            reduce_axis = []
            for dim in range(len(x.shape)):
                if dim >= self.begin_norm_axis:
                    reduce_axis.append(dim)
            mean = x.mean(dim=reduce_axis, keepdim=True)
            variance = x.var(dim=reduce_axis, unbiased=False, keepdim=True)
            params_shape = x.shape[self.begin_params_axis:]
            weight = self.weight
            bias = self.bias
            if len(mean.shape) == 1:
                nd_params_shape = [1] * len(x.shape)
                nd_params_shape[self.begin_norm_axis] = params_shape[0]
                mean = torch.reshape(mean, shape=nd_params_shape)
                variance = torch.reshape(variance, nd_params_shape)
                if (self.weight is not None and params_shape[0] == self.weight.nelement()):
                    weight = torch.reshape(self.weight, shape=nd_params_shape)
                if self.bias is not None and params_shape[0] == self.bias.nelement():
                    bias = torch.reshape(self.bias, shape=nd_params_shape)
            elif len(mean.shape) == len(x.shape):
                pass
            else:
                raise ValueError("shape of mean and variance should be 1D or has number of axes and x's")
            variance += self.eps
            normalized = (x - mean) * variance.rsqrt()
            if self.elementwise_affine:
                normalized = normalized * weight + bias
            return normalized
        else:
            if self.elementwise_affine:
                res = flow._C.layer_norm_affine(
                    x,
                    self.weight,
                    self.bias,
                    begin_norm_axis=self.begin_norm_axis,
                    begin_params_axis=self.begin_params_axis,
                    epsilon=self.eps,
                )
            else:
                res = flow._C.layer_norm(
                    x,
                    begin_norm_axis=self.begin_norm_axis,
                    begin_params_axis=self.begin_params_axis,
                    epsilon=self.eps,
                )
            return res

    def extra_repr(self) -> str:
        return "{normalized_shape}, eps={eps}, elementwise_affine={elementwise_affine}".format(**self.__dict__)
