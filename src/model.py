import torch
import torch.nn as nn
import torch.nn.functional as F
from xformers.components import MultiHeadDispatch, build_attention
from typing import Tuple
import math
from configuration import Configuration


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
        self.linear = nn.Linear(dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input, gate_input = self.linear(x).chunk(2, dim=-1)
        gate = self.swish(gate_input)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        src_ln = self.attention_norm(src)
        attn_output = self.attn(src_ln)
        src_and_attn = self.ffn_norm(src + attn_output)
        src_and_attn_and_ffwd = src_and_attn + self.ffn(src_and_attn)
        return src_and_attn_and_ffwd


class DecoderOnlyTransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, ffn_size,
                 input_dropout, attention_head_dropout,  attention_sublayer_dropout, ffn_sublayer_dropout,
                 num_heads, context_size, num_layers, num_actions, num_classes, device):
        super(DecoderOnlyTransformerModel, self).__init__()
        self.d_model = d_model
        self.example_input_embed = nn.Embedding(num_embeddings=vocab_size,
                                                embedding_dim=d_model)
        self.current_state_embed = nn.Embedding(num_embeddings=vocab_size,
                                                embedding_dim=d_model)
        self.attended_example_input_embed = nn.Embedding(num_embeddings=vocab_size,
                                                         embedding_dim=d_model)
        self.attended_current_state_embed = nn.Embedding(num_embeddings=vocab_size,
                                                         embedding_dim=d_model)

        self.dropout_1 = nn.Dropout(input_dropout)
        modules = [NonCausalSelfAttentionTransformerBlock(
            d_model, ffn_size, num_heads,
            attention_head_dropout,  attention_sublayer_dropout, ffn_sublayer_dropout,
            context_size, device) for _ in range(num_layers)]

        self.blocks = nn.Sequential(*modules)
        self.norm = nn.RMSNorm(d_model)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        example_input, current_state, attended_example_input, attended_current_state = x
        x_example_input = self.example_input_embed(
            example_input)
        x_current_state = self.current_state_embed(
            current_state)
        x_attended_example_input = self.attended_example_input_embed(
            attended_example_input)
        x_attended_current_state = self.attended_current_state_embed(
            attended_current_state)
        x = torch.cat([x_example_input,
                       x_current_state,
                       x_attended_example_input,
                      x_attended_current_state], dim=1)
        x = x / math.sqrt(self.d_model)
        # We use Dropout after computing embedding.
        # See File:Full GPT architecture.svg
        # https://en.wikipedia.org/wiki/File:Full_GPT_architecture.svg
        embed_drop = self.dropout_1(x)
        transformed = self.blocks(embed_drop)
        transformed_ln = self.norm(transformed)

        return transformed_ln


class ActionValueNetworkModel(nn.Module):
    """
    Predict action values given a state.
    """

    def __init__(self, config: Configuration, device: torch.device):
        super(ActionValueNetworkModel, self).__init__()
        self.__base_model = DecoderOnlyTransformerModel(
            config.vocab_size, config.d_model, config.d_ff,
            config.input_dropout, config.attention_head_dropout, config.attention_sublayer_dropout, config.ffn_sublayer_dropout,
            config.num_heads, config.context_size, config.num_layers, config.num_actions, config.num_classes, device)
        d_model = config.d_model
        num_actions = config.num_actions
        num_classes = config.num_classes
        self.classifier = nn.Linear(
            in_features=d_model, out_features=num_actions * num_classes)
        self.num_actions = num_actions
        self.num_classes = num_classes

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        hidden = self.__base_model(x)
        # [batch_size, context_size, d_model] -> [batch_size, context_size, num_actions*num_classes]
        logits = self.classifier(hidden)
        # [batch_size, context_size, num_actions*num_classes] -> [batch_size, num_actions*num_classes]
        mean_logits = logits.mean(dim=1)
        batch_size = mean_logits.shape[0]
        # [batch_size, num_actions*num_classes] -> [batch_size, num_actions, num_classes]
        action_mean_logits = mean_logits.view(
            batch_size, self.num_actions, self.num_classes)
        log_softmax_output = F.log_softmax(action_mean_logits, dim=-1)
        return log_softmax_output


class PolicyNetworkModel(nn.Module):
    """
    Predict action probabilities given a state.
    """

    def __init__(self, config: Configuration, device: torch.device):
        super(PolicyNetworkModel, self).__init__()
        self.__base_model = DecoderOnlyTransformerModel(
            config.vocab_size, config.d_model, config.d_ff,
            config.input_dropout, config.attention_head_dropout, config.attention_sublayer_dropout, config.ffn_sublayer_dropout,
            config.num_heads, config.context_size, config.num_layers, config.num_actions, config.num_classes, device)
        d_model = config.d_model
        num_actions = config.num_actions
        self.classifier = nn.Linear(
            in_features=d_model, out_features=num_actions)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        hidden = self.__base_model(x)
        # [batch_size, context_size, d_model] -> [batch_size, context_size, num_actions]
        logits = self.classifier(hidden)
        # [batch_size, context_size, num_actions*num_classes] -> [batch_size, num_actions]
        mean_logits = logits.mean(dim=1)
        log_softmax_output = F.log_softmax(mean_logits, dim=-1)
        return log_softmax_output
