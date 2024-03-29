# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

# Standard library imports
from typing import Optional, Tuple, List

# Third-party imports
from mxnet import autograd
import mxnet as mx
import numpy as np

# First-party imports
from gluonts.model.common import Tensor
from gluonts.core.component import validated

# Relative imports
from . import bijection as bij
from .distribution import Distribution, getF


class TransformedDistribution(Distribution):
    r"""
    A distribution obtained by applying a sequence of transformations on top
    of a base distribution.
    """

    @validated()
    def __init__(
        self, base_distribution: Distribution, transforms: List[bij.Bijection]
    ) -> None:
        self.base_distribution = base_distribution
        self.transforms = transforms
        self.is_reparameterizable = self.base_distribution.is_reparameterizable

        # use these to cache shapes and avoid recomputing all steps
        # the reason we cannot do the computations here directly
        # is that this constructor would fail in mx.symbol mode
        self._event_dim: Optional[int] = None
        self._event_shape: Optional[Tuple] = None
        self._batch_shape: Optional[Tuple] = None

    @property
    def event_dim(self):
        if self._event_dim is None:
            self._event_dim = max(
                [self.base_distribution.event_dim]
                + [t.event_dim for t in self.transforms]
            )
        assert isinstance(self._event_dim, int)
        return self._event_dim

    @property
    def batch_shape(self) -> Tuple:
        if self._batch_shape is None:
            shape = (
                self.base_distribution.batch_shape
                + self.base_distribution.event_shape
            )
            self._batch_shape = shape[: len(shape) - self.event_dim]
        assert isinstance(self._batch_shape, tuple)
        return self._batch_shape

    @property
    def event_shape(self) -> Tuple:
        if self._event_shape is None:
            shape = (
                self.base_distribution.batch_shape
                + self.base_distribution.event_shape
            )
            self._event_shape = shape[len(shape) - self.event_dim :]
        assert isinstance(self._event_shape, tuple)
        return self._event_shape

    def sample(self, num_samples: Optional[int] = None , dtype = np.float32 ) -> Tensor:
        with autograd.pause():
            s = self.base_distribution.sample(num_samples=num_samples)
            for t in self.transforms:
                s = t.f(s)
            return s

    def sample_rep(self, num_samples: Optional[int] = None) -> Tensor:
        s = self.base_distribution.sample_rep()
        for t in self.transforms:
            s = t.f(s)
        return s

    def log_prob(self, y: Tensor) -> Tensor:
        F = getF(y)
        lp = 0.0
        x = y
        for t in self.transforms[::-1]:
            x = t.f_inv(y)
            ladj = t.log_abs_det_jac(x, y)
            lp -= sum_trailing_axes(F, ladj, self.event_dim - t.event_dim)
            y = x

        return self.base_distribution.log_prob(x) + lp

    def cdf(self, y: Tensor) -> Tensor:
        x = y
        sign = 1.0
        for t in self.transforms[::-1]:
            x = t.f_inv(x)
            sign = sign * t.sign
        f = self.base_distribution.cdf(x)
        return sign * (f - 0.5) + 0.5

    def quantile(self, level: Tensor) -> Tensor:
        F = getF(level)

        sign = 1.0
        for t in self.transforms:
            sign = sign * t.sign

        if not isinstance(sign, (mx.nd.NDArray, mx.sym.Symbol)):
            sign = sign + level.zeros_like()

        cond = F.broadcast_greater(sign, sign.zeros_like())
        level = F.broadcast_mul(cond, level) + F.broadcast_mul(
            1.0 - cond, 1.0 - level
        )

        q = self.base_distribution.quantile(level)
        for t in self.transforms:
            q = t.f(q)
        return q


def sum_trailing_axes(F, x: Tensor, k: int) -> Tensor:
    for _ in range(k):
        x = F.sum(x, axis=-1)
    return x
