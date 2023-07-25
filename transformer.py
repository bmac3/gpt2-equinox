import equinox as eqx
import haliax as hax
from haliax import Axis, NamedArray
import haliax.nn as hnn
from jax import random
from jaxtyping import Float, Int

from src.models.utils import sequential
from src.models.gpt.attention import MultiHeadAttention
from src.models.gpt.mlp import MLP


class TransformerLayer(eqx.Module):
    ln1: hnn.LayerNorm
    attn: MultiHeadAttention
    ln2: hnn.LayerNorm
    mlp: MLP
    
    def __call__(
            self, 
            Pos: Axis, 
            embeddings: Float[NamedArray, 'position embedding'], 
            mask: Int[NamedArray, 'position']
        ):

        layers = [
            lambda x: x + self.attn(Pos, self.ln1(x), mask),
            lambda x: x + self.mlp(self.ln2(x))
        ]
        
        return sequential(layers, embeddings)
        
    @staticmethod
    def init(config, *, prng_key):
        Embed = Axis('embedding', config.embedding_size)

        attn_prng, mlp_prng = random.split(prng_key, 2)

        ln1 = hnn.LayerNorm.init(Embed, eps=config.layer_norm_eps)
        attn = MultiHeadAttention.init(config, prng_key=attn_prng)
        ln2 = hnn.LayerNorm.init(Embed, eps=config.layer_norm_eps)
        mlp = MLP.init(config, prng_key=mlp_prng)

        return TransformerLayer(ln1, attn, ln2, mlp)
    

class TransformerStack(eqx.Module):
    stack: TransformerLayer
    Layers: Axis = eqx.static_field()

    def __call__(self, Pos, embeddings, mask):

        def do_block(embeds, layer):
            return layer(Pos, embeds, mask)

        return hax.fold(do_block, self.Layers)(embeddings, self.stack)

    @staticmethod
    def init(config, *, prng_key):
        Layers = Axis('layer', config.num_layers)
        stacked = hax.vmap(TransformerLayer.init, Layers)(config, prng_key=random.split(prng_key, Layers.size))
        return TransformerStack(stacked, Layers)
