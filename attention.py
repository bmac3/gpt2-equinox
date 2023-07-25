from typing import Optional

import equinox as eqx
import haliax as hax
from haliax import Axis, NamedArray
import haliax.nn as hnn
import jax.numpy as jnp
from jax import random
from jaxtyping import Float, Int


def dot_product_attention_weights(
        Embed: Axis,
        KPos: Axis,
        query: Float[NamedArray, 'embedding'],
        keys: Float[NamedArray, 'key_position embedding'],
        mask: Optional[Int[NamedArray, 'key_position']] = None,
        bias: Optional[Float[NamedArray, 'key_position']] = None,
    ):
    query = query / jnp.sqrt(Embed.size)
    weights = hax.dot(Embed, query, keys)

    if bias is not None:
        weights = weights + bias
    if mask is not None:
        weights = hax.where(mask, weights, -1e9)

    weights = hnn.softmax(weights, KPos)
    return weights


class SelfAttention(eqx.Module):
    query_read_in: hnn.Linear
    key_read_in: hnn.Linear
    value_read_in: hnn.Linear
    QKEmbed: Axis = eqx.static_field()

    def __call__(
            self, 
            Pos: Axis, 
            embeddings: Float[NamedArray, 'position embedding'], 
            mask: Int[NamedArray, 'position']
        ):
        QPos = Pos.alias('query_position')
        KVPos = Pos.alias('key_value_position')

        mask = mask.rename({Pos: KVPos})
        mask = self.reformat_mask(QPos, KVPos, mask)

        queries = self.query_read_in(embeddings)
        keys = self.key_read_in(embeddings)
        values = self.value_read_in(embeddings)

        queries = queries.rename({Pos: QPos})
        keys = keys.rename({Pos: KVPos})
        values = values.rename({Pos: KVPos})

        attn_weights = hax.vmap(dot_product_attention_weights, QPos)(self.QKEmbed, KVPos, queries, keys, mask)
        attn = hax.dot(KVPos, attn_weights, values)
        attn = attn.rename({QPos: Pos})
        return attn

    @staticmethod
    def reformat_mask(
            QPos: Axis, 
            KVPos: Axis, 
            mask: Int[NamedArray, 'key_value_position']
        ):
        causal_mask = hnn.attention.causal_mask(QPos, KVPos)
        expanded_mask = mask.broadcast_axis(QPos)
        return hnn.attention.combine_masks_and(expanded_mask, causal_mask)

    @staticmethod
    def init(config, *, prng_key):
        Embed = Axis('embedding', config.embedding_size)
        QKEmbed = Axis('query_key_embedding', config.query_key_embedding_size)
        VEmbed = Axis('value_embedding', config.value_embedding_size)

        query_prng, key_prng, value_prng = random.split(prng_key, 3)
        
        query_read_in = hnn.Linear.init(In=Embed, Out=QKEmbed, key=query_prng)
        key_read_in = hnn.Linear.init(In=Embed, Out=QKEmbed, key=key_prng)
        value_read_in = hnn.Linear.init(In=Embed, Out=VEmbed, key=value_prng)

        return SelfAttention(query_read_in, key_read_in, value_read_in, QKEmbed)


class MultiHeadAttention(eqx.Module):
    attn_heads: SelfAttention
    write_out: hnn.Linear
    Head: Axis = eqx.static_field()
    
    def __call__(
            self,
            Pos: Axis,
            embeddings: Float[NamedArray, 'position embedding'],
            mask: Int[NamedArray, 'position']
        ):

        def do_block(attn_head):
            return attn_head(Pos, embeddings, mask)

        attn_out = hax.vmap(do_block, self.Head)(self.attn_heads)
        return self.write_out(attn_out)

    @staticmethod
    def init(config, *, prng_key):
        Head = Axis('head', config.num_heads)
        VEmbed = Axis('value_embedding', config.value_embedding_size)
        Embed = Axis('embedding', config.embedding_size)

        attn_prng, write_out_prng = random.split(prng_key, 2)

        attn_heads = hax.vmap(SelfAttention.init, Head)(config, prng_key=random.split(attn_prng, Head.size))
        write_out = hnn.Linear.init(In=(Head, VEmbed), Out=Embed, key=write_out_prng)

        return MultiHeadAttention(attn_heads, write_out, Head)
    