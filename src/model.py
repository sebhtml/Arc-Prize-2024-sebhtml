import torch
import torch.nn as nn
import xformers.ops as xops
from xformers.components.positional_embedding import RotaryEmbedding
from xformers.components import MultiHeadDispatch, build_attention

from torch import nn
import torch
import math


class SwiGLU(nn.Module):
    """
    swiGLU = (x * U) * swish(x * V)
    Note that in PyTorch, the name of Swish is SILU

    See https://arxiv.org/pdf/2002.05202
    See https://jcarlosroldan.com/post/348/what-is-swiglu
    """

    def __init__(self, dim):
        super().__init__()
        self.swish = nn.SiLU()
        self.input_proj = nn.Linear(dim, dim, bias=False)
        self.gate_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        input = self.input_proj(x)
        gate = self.swish(self.gate_proj(x))
        return input * gate


class FeedForward(nn.Module):
    """
    Feed-forward network for a transformer block
    See https://en.wikipedia.org/wiki/GPT-1#/media/File:Full_GPT_architecture.svg
    """

    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.swiglu = SwiGLU(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.swiglu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


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


class CustomAttention(nn.Module):
    """
    This custom attention uses multi-head attention with non-causal self-attention and rotary embeddings.
    See https://en.wikipedia.org/wiki/GPT-1#/media/File:Full_GPT_architecture.svg
    """

    def __init__(self, num_heads, d_model,
                 attention_head_dropout,  attention_sublayer_dropout,
                 context_size, device):
        super().__init__()
        my_config = {
            "name": "scaled_dot_product",
            "dropout": attention_head_dropout,
            "seq_len": context_size,
            "causal": False,
        }
        attention = build_attention(my_config)
        self.attn = MultiHeadDispatch(
            seq_len=context_size,
            dim_model=d_model,
            residual_dropout=attention_sublayer_dropout,
            num_heads=num_heads,
            attention=attention,
            use_rotary_embeddings=True,
        ).to(device)

    def forward(self, x):
        # Self-attention
        return self.attn(query=x, key=x, value=x)


class NonCausalSelfAttentionTransformerBlock(nn.Module):
    def __init__(self, d_model, ffn_size, num_heads,
                 attention_head_dropout,  attention_sublayer_dropout, ffn_sublayer_dropout,
                 context_size, device):
        super(NonCausalSelfAttentionTransformerBlock, self).__init__()

        self.attn = CustomAttention(
            num_heads, d_model,
            attention_head_dropout,  attention_sublayer_dropout,
            context_size, device)

        self.ffn = FeedForward(d_model, ffn_size, ffn_sublayer_dropout)
        self.attention_norm = nn.RMSNorm(d_model)
        self.ffn_norm = nn.RMSNorm(d_model)

    def forward(self, src):
        src_ln = self.attention_norm(src)
        attn_output = self.attn(src_ln)
        src_and_attn = self.ffn_norm(src + attn_output)
        src_and_attn_and_ffwd = src_and_attn + self.ffn(src_and_attn)
        return src_and_attn_and_ffwd


class DecoderOnlyTransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, ffn_size,
                 input_dropout, attention_head_dropout,  attention_sublayer_dropout, ffn_sublayer_dropout,
                 num_heads, context_size, num_layers, num_classes, device):
        super(DecoderOnlyTransformerModel, self).__init__()
        self.d_model = d_model
        self.current_state_embed = nn.Embedding(num_embeddings=vocab_size,
                                                embedding_dim=d_model)
        self.attended_example_input_embed = nn.Embedding(num_embeddings=vocab_size,
                                                         embedding_dim=d_model)
        self.attended_current_state_embed = nn.Embedding(num_embeddings=vocab_size,
                                                         embedding_dim=d_model)
        self.attended_action_embed = nn.Embedding(num_embeddings=vocab_size,
                                                  embedding_dim=d_model)

        self.dropout_1 = nn.Dropout(input_dropout)
        modules = [NonCausalSelfAttentionTransformerBlock(
            d_model, ffn_size, num_heads,
            attention_head_dropout,  attention_sublayer_dropout, ffn_sublayer_dropout,
            context_size, device) for _ in range(num_layers)]

        self.blocks = nn.Sequential(*modules)
        self.norm = nn.RMSNorm(d_model)

        self.gap = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Linear(
            in_features=d_model, out_features=num_classes)

    def forward(self, x):
        current_state, attended_example_input, attended_current_state, attended_action = x
        x_current_state = self.current_state_embed(
            current_state)
        x_attended_example_input = self.attended_example_input_embed(
            attended_example_input)
        x_attended_current_state = self.attended_current_state_embed(
            attended_current_state)
        x_attended_action = self.attended_action_embed(attended_action)
        x = torch.cat([x_current_state,
                       x_attended_example_input,
                      x_attended_current_state, x_attended_action], dim=1)
        x = x / math.sqrt(self.d_model)
        # We use Dropout after computing embedding.
        # See File:Full GPT architecture.svg
        # https://en.wikipedia.org/wiki/File:Full_GPT_architecture.svg
        embed_drop = self.dropout_1(x)
        transformed = self.blocks(embed_drop)
        transformed_ln = self.norm(transformed)
        last_hidden_state = transformed_ln
        output = self.gap(last_hidden_state.transpose(1, 2))
        output = output.squeeze(2)
        logits = self.classifier(output)
        return logits
