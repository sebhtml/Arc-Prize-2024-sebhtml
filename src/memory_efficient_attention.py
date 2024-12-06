import xformers.ops as xops
from xformers.components.positional_embedding import RotaryEmbedding
from torch import nn


class CustomMemoryEfficientAttention(nn.Module):
    """
    This custom attention uses multi-head attention with non-causal self-attention and rotary embeddings.

    See
    https://en.wikipedia.org/wiki/GPT-1#/media/File:Full_GPT_architecture.svg

    See
    Self-attention Does Not Need O(n2) Memory
    https://arxiv.org/abs/2112.05682

    See
    https://facebookresearch.github.io/xformers/components/ops.html

    See
    Rotary Embeddings: A Relative Revolution
    https://blog.eleuther.ai/rotary-embeddings/

    See 
    RoFormer: Enhanced Transformer with Rotary Position Embedding
    https://arxiv.org/abs/2104.09864
    """

    def __init__(self, num_heads, d_model,
                 attention_head_dropout,  attention_sublayer_dropout,
                 context_size, device):
        super().__init__()
        if d_model % num_heads != 0:
            raise Exception(
                f"d_model {d_model} is not a multiple of num_heads {num_heads}")
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        self.attention_head_dropout = attention_head_dropout
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.rotary_embedding = RotaryEmbedding(self.head_dim)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout_op = nn.Dropout(attention_sublayer_dropout)

    def forward(self, x):
        # Linear projections
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        batch_size = q.size()[0]

        # B = batch size
        # M = sequence length
        # D = embed dim
        # H = num heads
        # K = head dim

        # Split the last dimension in H heads for multi-head attention.
        # [B, M, D] -> [B, M, H, K]
        q = q.view(batch_size, -1, self.num_heads, self.head_dim)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim)

        # Do the rotary embedding thing to allow learning on sequences.
        # Rotary embedding needs the input shape (BATCH, HEADS, SEQ, EMB).
        # [B, M, H, K] -> [B, H, M, K]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        q, k = self.rotary_embedding(q, k)
        # [B, H, M, K] -> [B, M, H, K]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)

        # Input tensors must be in format [B, M, H, K]
        # No causal masking for non-causal attention.
        # Let the runtime decides which op to use.
        attention_output = xops.memory_efficient_attention(
            q,
            k,
            v,
            p=self.attention_head_dropout,
        )

        # Concat the H heads together.
        # [B, M, H, K] -> [B, M, D]
        attention_output = attention_output.view(batch_size, -1, self.d_model)

        # Do the linear projection of the output
        proj_attention = self.out_proj(attention_output)
        output = self.dropout_op(proj_attention)
        return output
