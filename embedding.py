import equinox as eqx
import haliax as hax
from haliax import Axis, NamedArray
import haliax.nn as hnn
from jaxtyping import Float, Int


class Embedding(eqx.Module):
    token_embeddings: hnn.Embedding
    position_embeddings: hnn.Embedding

    def embed(
            self, 
            Pos: Axis,
            input_ids: Int[NamedArray, 'position']
        ):
        position_ids = hax.arange(Pos)
        return self.token_embeddings.embed(input_ids) + self.position_embeddings.embed(position_ids)
    
    def unembed(
            self,
            embeddings: Float[NamedArray, 'position embedding']
        ):
        return self.token_embeddings.unembed(embeddings)
    
    @staticmethod
    def init(config, *, prng_key):
        Embed = Axis('embedding', config.embedding_size)
        Vocab = Axis('vocab', config.vocab_size)
        MaxPos = Axis('position', config.max_sequence_length)

        token_embeddings = hnn.Embedding.init(Vocab, Embed, key=prng_key, initializer_range=config.initializer_range)
        position_embeddings = hnn.Embedding.init(MaxPos, Embed, key=prng_key, initializer_range=config.initializer_range)

        return Embedding(token_embeddings, position_embeddings)
