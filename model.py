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

    def forward(self, query, key, value):
        return self.attn(query, key, value)


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
        # Self-attention
        # TODO pass only src_ln since we use a SelfAttentionMechanism
        attn_output = self.attn(
            query=src_ln, key=src_ln, value=src_ln)
        src_and_attn = self.ffn_norm(src + attn_output)
        src_and_attn_and_ffwd = src_and_attn + self.ffn(src_and_attn)
        return src_and_attn_and_ffwd


class DecoderOnlyTransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, ffn_size,
                 input_dropout, attention_head_dropout,  attention_sublayer_dropout, ffn_sublayer_dropout,
                 num_heads, context_size, num_layers, num_classes, device):
        super(DecoderOnlyTransformerModel, self).__init__()
        self.d_model = d_model
        self.input_embed = nn.Embedding(num_embeddings=vocab_size,
                                        embedding_dim=d_model)
        self.counter_embed = nn.Embedding(num_embeddings=vocab_size,
                                          embedding_dim=d_model)
        self.current_embed = nn.Embedding(num_embeddings=vocab_size,
                                          embedding_dim=d_model)
        self.action_embed = nn.Embedding(num_embeddings=vocab_size,
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
        input_state, current_state, action = x
        x_input_state = self.input_embed(input_state)
        x_current_state = self.current_embed(current_state)
        x_action = self.action_embed(action)
        x = torch.cat([x_input_state, x_current_state, x_action], dim=1)
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
