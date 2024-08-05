
from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
import torch.nn.functional as F
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.attention_processor import Attention
import inspect
from functools import partial
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn


class IPJointAttnProcessor2_0 (nn.Module):
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self, hidden_size,cross_attention_dim, scale=1.0, num_tokens=8,num_heads=12):
        super().__init__()

        self.hidden_size = hidden_size
        self.scale = scale
        self.num_tokens = num_tokens
        self.num_heads = num_heads

        self.to_k_ip = nn.Linear(cross_attention_dim, cross_attention_dim, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim, cross_attention_dim, bias=False)

        self.add_k_proj_ip = nn.Linear(cross_attention_dim, cross_attention_dim, bias=False)
        self.add_v_proj_ip = nn.Linear(cross_attention_dim, cross_attention_dim, bias=False)

        self.proj_ip = nn.Linear(cross_attention_dim, cross_attention_dim, bias=False)
        self.proj_encoder_hidden_states_ip = nn.Linear(cross_attention_dim, cross_attention_dim, bias=False)
        self.proj_encoder_passthrough_ip = nn.Linear(cross_attention_dim, cross_attention_dim, bias=False)
        #zero init proj
        nn.init.zeros_(self.proj_ip.weight)
        nn.init.zeros_(self.proj_encoder_hidden_states_ip.weight)
        nn.init.eye_(self.proj_encoder_passthrough_ip.weight)


    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        ip_encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states
        target_seq_length = hidden_states.shape[1]
        ip_encoder_hidden_states_original = ip_encoder_hidden_states
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        context_input_ndim = encoder_hidden_states.ndim
        if context_input_ndim == 4:
            batch_size, channel, height, width = encoder_hidden_states.shape
            encoder_hidden_states = encoder_hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size = encoder_hidden_states.shape[0]

        # Original attention mechanism (unchanged)
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        query_combined = torch.cat([query, encoder_hidden_states_query_proj], dim=1)
        key_combined = torch.cat([key, encoder_hidden_states_key_proj], dim=1)
        value_combined = torch.cat([value, encoder_hidden_states_value_proj], dim=1)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query_combined = query_combined.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key_combined = key_combined.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value_combined = value_combined.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(query_combined, key_combined, value_combined, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states, encoder_hidden_states = (
            hidden_states[:, :residual.shape[1]],
            hidden_states[:, residual.shape[1]:],
        )

        #oops, this should be refering to the original hidden_states, not the new ones
        #ip key and value
        ip_key = self.to_k_ip(hidden_states)
        ip_value = self.to_v_ip(hidden_states)

        # Prepare joint query from hidden_states and encoder_hidden_states
        #ip_encoder_hidden_states_query_proj = attn.add_q_proj_ip(ip_encoder_hidden_states)
        ip_encoder_hidden_states_key_proj = self.add_k_proj_ip(ip_encoder_hidden_states)
        ip_encoder_hidden_states_value_proj = self.add_v_proj_ip(ip_encoder_hidden_states)

        #ip_query_combined = torch.cat([ip_query, ip_encoder_hidden_states_query_proj], dim=1)
        ip_key_combined = torch.cat([ip_key, ip_encoder_hidden_states_key_proj], dim=1)
        ip_value_combined = torch.cat([ip_value, ip_encoder_hidden_states_value_proj], dim=1)


        # Perform attention
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        #ip_query_combined = ip_query_combined.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        ip_key_combined = ip_key_combined.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        ip_value_combined = ip_value_combined.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        ip_hidden_states = F.scaled_dot_product_attention(query_combined, ip_key_combined, ip_value_combined, dropout_p=0.0, is_causal=False)
        ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        ip_hidden_states = ip_hidden_states.to(query.dtype)

        
        # Split the attention outputs
        ip_hidden_states, ip_encoder_hidden_states = (
            ip_hidden_states[:, :residual.shape[1]],
            ip_hidden_states[:, residual.shape[1]:],
        )


        #ip proj out
        ip_hidden_states = self.proj_ip(ip_hidden_states)
        ip_encoder_hidden_states = self.proj_encoder_hidden_states_ip(ip_encoder_hidden_states)
        ip_encoder_hidden_states_original = self.proj_encoder_passthrough_ip(ip_encoder_hidden_states_original)

        hidden_states = hidden_states + (self.scale * ip_hidden_states)
        encoder_hidden_states = encoder_hidden_states + (self.scale * ip_encoder_hidden_states)

        # Linear projection and dropout (unchanged)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if not attn.context_pre_only:
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if context_input_ndim == 4:
            encoder_hidden_states = encoder_hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        return hidden_states, encoder_hidden_states,ip_encoder_hidden_states_original


class JointAttnProcessor2_0 (nn.Module):
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self,num_tokens):
        super().__init__()
        self.num_tokens = num_tokens

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        ip_encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        
        residual = hidden_states

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        context_input_ndim = encoder_hidden_states.ndim
        if context_input_ndim == 4:
            batch_size, channel, height, width = encoder_hidden_states.shape
            encoder_hidden_states = encoder_hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size = encoder_hidden_states.shape[0]

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # `context` projections.
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        # attention
        query = torch.cat([query, encoder_hidden_states_query_proj], dim=1)
        key = torch.cat([key, encoder_hidden_states_key_proj], dim=1)
        value = torch.cat([value, encoder_hidden_states_value_proj], dim=1)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # Split the attention outputs.
        hidden_states, encoder_hidden_states = (
            hidden_states[:, : residual.shape[1]],
            hidden_states[:, residual.shape[1] :],
        )

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        if not attn.context_pre_only:
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if context_input_ndim == 4:
            encoder_hidden_states = encoder_hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        return hidden_states, encoder_hidden_states,ip_encoder_hidden_states
