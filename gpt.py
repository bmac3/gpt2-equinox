from dataclasses import dataclass

import equinox as eqx
from haliax import Axis, NamedArray
from jax import random
from jaxtyping import Int


from .utils import sequential
from .embedding import Embedding
from .transformer import TransformerStack
from .lm_transform import LMTransform


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    embedding_size: int = 384
    query_key_embedding_size: int = 64
    value_embedding_size: int = 64
    num_heads: int = 6
    num_layers: int = 6
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    intermediate_size: int = 1536
    max_sequence_length: int = 1024


class GPT(eqx.Module):
    embedding: Embedding
    transformer_layers: TransformerStack
    lm_transform: LMTransform

    def __call__(
            self, 
            Pos: Axis,
            token_ids: Int[NamedArray, 'position'],
            mask: Int[NamedArray, 'position']
        ):

        layers = [
            lambda x: self.embedding.embed(Pos, x),
            lambda x: self.transformer_layers(Pos, x, mask),
            self.lm_transform,
            self.embedding.unembed
        ]

        return sequential(layers, token_ids)

    @staticmethod
    def init(
            config, 
            *, 
            prng_key
        ):
        embed_prng, transformer_prng, lm_transform_prng = random.split(prng_key, 3)
        
        embedding = Embedding.init(config, prng_key=embed_prng)
        transformer_layers = TransformerStack.init(config, prng_key=transformer_prng)
        lm_transform = LMTransform.init(config, prng_key=lm_transform_prng)

        return GPT(embedding, transformer_layers, lm_transform)
    