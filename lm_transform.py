import equinox as eqx
from haliax import Axis
import haliax.nn as hnn

from .utils import sequential


class DuplicateAxisLinear(eqx.Module):
    linear: hnn.Linear
    axis: Axis = eqx.static_field()
    In: Axis = eqx.static_field()
    Out: Axis = eqx.static_field()

    def __call__(self, x):
        layers = [
            lambda x: x.rename({self.axis: self.In}),
            self.linear,
            lambda x: x.rename({self.Out: self.axis})
        ]
        return sequential(layers, x)

    @staticmethod
    def init(axis, *, prng_key):
        In = axis.alias('in')
        Out = axis.alias('out')
        linear = hnn.Linear.init(In=In, Out=Out, key=prng_key)
        return DuplicateAxisLinear(linear, axis, In, Out)


class LMTransform(eqx.Module):
    linear: hnn.Linear
    ln: hnn.LayerNorm

    def __call__(self, embeddings):

        layers = [
            self.ln,
            self.linear,
            hnn.gelu,
        ]

        return sequential(layers, embeddings)

    @staticmethod
    def init(config, *, prng_key):
        Embed = Axis('embedding', config.embedding_size)

        linear = DuplicateAxisLinear.init(Embed, prng_key=prng_key)
        ln = hnn.LayerNorm.init(Embed, eps=config.layer_norm_eps)

        return LMTransform(linear, ln)
    