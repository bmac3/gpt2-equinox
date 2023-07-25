import equinox as eqx
from haliax import Axis, NamedArray
import haliax.nn as hnn
from jax import random
from jaxtyping import Float

from .utils import sequential


class MLP(eqx.Module):
    linear_in: hnn.Linear
    linear_out: hnn.Linear

    def __call__(
            self, 
            embedding: Float[NamedArray, 'embedding']
        ):

        layers = [
            self.linear_in,
            hnn.gelu,
            self.linear_out
        ]
        
        return sequential(layers, embedding)

    @staticmethod
    def init(config, *, prng_key):
        Embed = Axis('embedding', config.embedding_size)
        Intermediate = Axis('intermediate', config.intermediate_size)

        prng_in, prng_out = random.split(prng_key, 2)

        linear_in = hnn.Linear.init(In=Embed, Out=Intermediate, key=prng_in)
        linear_out = hnn.Linear.init(In=Intermediate, Out=Embed, key=prng_out)

        return MLP(linear_in, linear_out)
    