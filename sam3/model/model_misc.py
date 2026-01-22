# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""Various utility models"""

import copy
import math
import weakref
from collections.abc import Iterator
from contextlib import AbstractContextManager
from enum import auto, Enum
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing_extensions import override


def inverse_sigmoid(x, eps=1e-3):
    """
    The inverse function for sigmoid activation function.
    Note: It might face numberical issues with fp16 small eps.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class MultiheadAttentionWrapper(nn.MultiheadAttention):
    def forward(self, *args, **kwargs):
        kwargs["need_weights"] = False
        return super().forward(*args, **kwargs)


class ExportFriendlyMultiheadAttention(nn.Module):
    """MultiheadAttention using F.scaled_dot_product_attention for torch.export compatibility.

    The standard nn.MultiheadAttention uses F.multi_head_attention_forward which has
    internal guards on sequence length that fail with dynamic shapes in torch.export.
    This implementation uses F.scaled_dot_product_attention directly to avoid those guards.

    Why this is needed:
    -------------------
    When exporting with dynamic shapes (e.g., variable image H/W), PyTorch's
    F.multi_head_attention_forward creates guards like `Eq(seq_len, 5184)` which
    fail during torch.export because the sequence length is symbolic.

    The guard happens in shape validation code (checking attn_mask dimensions)
    BEFORE scaled_dot_product_attention is even called, so using
    `sdpa_kernel([SDPBackend.MATH])` alone doesn't help.

    This class bypasses F.multi_head_attention_forward entirely by:
    1. Manually projecting Q, K, V
    2. Calling F.scaled_dot_product_attention directly
    3. Avoiding all shape validation guards

    Related PyTorch issues:
    - https://github.com/pytorch/pytorch/issues/170127
    - https://github.com/pytorch/pytorch/issues/124502
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first = batch_first
        self.dropout = dropout

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Combined QKV projection for efficiency
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter("in_proj_bias", None)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            nn.init.zeros_(self.in_proj_bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    @classmethod
    def from_nn_mha(cls, mha: nn.MultiheadAttention) -> "ExportFriendlyMultiheadAttention":
        """Create an ExportFriendlyMultiheadAttention from an nn.MultiheadAttention.

        Copies all weights from the source module.

        Args:
            mha: Source nn.MultiheadAttention module

        Returns:
            New ExportFriendlyMultiheadAttention with copied weights
        """
        # Create new instance with same configuration
        new_mha = cls(
            embed_dim=mha.embed_dim,
            num_heads=mha.num_heads,
            dropout=mha.dropout,
            bias=mha.in_proj_bias is not None,
            batch_first=mha.batch_first,
        )

        # Copy weights
        with torch.no_grad():
            new_mha.in_proj_weight.copy_(mha.in_proj_weight)
            if mha.in_proj_bias is not None:
                new_mha.in_proj_bias.copy_(mha.in_proj_bias)
            new_mha.out_proj.weight.copy_(mha.out_proj.weight)
            if mha.out_proj.bias is not None:
                new_mha.out_proj.bias.copy_(mha.out_proj.bias)

        return new_mha

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        need_weights: bool = False,
    ) -> tuple[Tensor, None]:
        """Forward pass using scaled dot product attention.

        Args:
            query: (L, N, E) or (N, L, E) if batch_first
            key: (S, N, E) or (N, S, E) if batch_first
            value: (S, N, E) or (N, S, E) if batch_first
            key_padding_mask: (N, S) where True means ignore
            attn_mask: (L, S) or (N*num_heads, L, S)
            need_weights: Ignored, always returns None for weights

        Returns:
            attn_output: (L, N, E) or (N, L, E) if batch_first
            attn_weights: Always None (for compatibility)
        """
        # Convert to batch_first format for SDPA
        if not self.batch_first:
            query = query.transpose(0, 1)  # (N, L, E)
            key = key.transpose(0, 1)  # (N, S, E)
            value = value.transpose(0, 1)  # (N, S, E)

        batch_size, tgt_len, _ = query.shape
        src_len = key.shape[1]

        # Project Q, K, V using combined weight
        # For cross-attention, we project Q from query and K,V from key/value
        q = F.linear(
            query,
            self.in_proj_weight[: self.embed_dim],
            self.in_proj_bias[: self.embed_dim] if self.in_proj_bias is not None else None,
        )
        k = F.linear(
            key,
            self.in_proj_weight[self.embed_dim : 2 * self.embed_dim],
            self.in_proj_bias[self.embed_dim : 2 * self.embed_dim]
            if self.in_proj_bias is not None
            else None,
        )
        v = F.linear(
            value,
            self.in_proj_weight[2 * self.embed_dim :],
            self.in_proj_bias[2 * self.embed_dim :] if self.in_proj_bias is not None else None,
        )

        # Reshape to (N, num_heads, L/S, head_dim)
        q = q.view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Handle attention mask
        # SDPA expects: (N, num_heads, L, S) or (L, S) broadcastable
        # Input attn_mask is (L, S) or (N*num_heads, L, S)
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                # (L, S) -> expand for SDPA
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, L, S)
            elif attn_mask.dim() == 3:
                # (N*num_heads, L, S) -> (N, num_heads, L, S)
                attn_mask = attn_mask.view(batch_size, self.num_heads, tgt_len, src_len)

        # Handle key_padding_mask
        # SDPA expects additive mask, True = -inf
        if key_padding_mask is not None:
            # (N, S) -> (N, 1, 1, S)
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            # Convert bool mask to additive mask
            key_padding_mask = key_padding_mask.to(dtype=q.dtype)
            key_padding_mask = key_padding_mask.masked_fill(key_padding_mask.bool(), float("-inf"))

            if attn_mask is not None:
                attn_mask = attn_mask + key_padding_mask
            else:
                attn_mask = key_padding_mask

        # Use scaled dot product attention
        dropout_p = self.dropout if self.training else 0.0
        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p
        )

        # Reshape back: (N, num_heads, L, head_dim) -> (N, L, E)
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.embed_dim)
        )

        # Output projection
        attn_output = self.out_proj(attn_output)

        # Convert back to seq_first if needed
        if not self.batch_first:
            attn_output = attn_output.transpose(0, 1)  # (L, N, E)

        return attn_output, None


def replace_mha_with_export_friendly(
    module: nn.Module,
    verbose: bool = False,
) -> int:
    """Replace all nn.MultiheadAttention modules with ExportFriendlyMultiheadAttention.

    This enables torch.export with dynamic shapes by bypassing
    F.multi_head_attention_forward which has guards on sequence length.

    Recursively traverses the module tree and replaces:
    - nn.MultiheadAttention
    - MultiheadAttentionWrapper (subclass of nn.MultiheadAttention)

    Args:
        module: Root module to process (modified in-place)
        verbose: If True, print each replacement

    Returns:
        Number of modules replaced
    """
    count = 0

    for name, child in list(module.named_children()):
        if isinstance(child, nn.MultiheadAttention):
            # Replace with export-friendly version
            new_child = ExportFriendlyMultiheadAttention.from_nn_mha(child)
            setattr(module, name, new_child)
            count += 1
            if verbose:
                print(
                    f"Replaced {name}: {type(child).__name__} -> ExportFriendlyMultiheadAttention"
                )
        else:
            # Recurse into children
            count += replace_mha_with_export_friendly(child, verbose=verbose)

    return count


class DotProductScoring(torch.nn.Module):
    def __init__(
        self,
        d_model,
        d_proj,
        prompt_mlp=None,
        clamp_logits=True,
        clamp_max_val=12.0,
    ):
        super().__init__()
        self.d_proj = d_proj
        assert isinstance(prompt_mlp, torch.nn.Module) or prompt_mlp is None
        self.prompt_mlp = prompt_mlp  # an optional MLP projection for prompt
        self.prompt_proj = torch.nn.Linear(d_model, d_proj)
        self.hs_proj = torch.nn.Linear(d_model, d_proj)
        self.scale = float(1.0 / np.sqrt(d_proj))
        self.clamp_logits = clamp_logits
        if self.clamp_logits:
            self.clamp_max_val = clamp_max_val

    def mean_pool_text(self, prompt, prompt_mask):
        # is_valid has shape (seq, bs, 1), where 1 is valid and 0 is padding
        is_valid = (~prompt_mask).float().permute(1, 0)[..., None]
        # num_valid has shape (bs, 1)
        num_valid = torch.clamp(torch.sum(is_valid, dim=0), min=1.0)
        # mean pool over all the valid tokens -- pooled_prompt has shape (bs, proj_dim)
        pooled_prompt = (prompt * is_valid).sum(dim=0) / num_valid
        return pooled_prompt

    def forward(self, hs, prompt, prompt_mask):
        # hs has shape (num_layer, bs, num_query, d_model)
        # prompt has shape (seq, bs, d_model)
        # prompt_mask has shape (bs, seq), where 1 is valid and 0 is padding
        assert hs.dim() == 4 and prompt.dim() == 3 and prompt_mask.dim() == 2

        # apply MLP on prompt if specified
        if self.prompt_mlp is not None:
            prompt = self.prompt_mlp(prompt)

        # first, get the mean-pooled version of the prompt
        pooled_prompt = self.mean_pool_text(prompt, prompt_mask)

        # then, project pooled_prompt and hs to d_proj dimensions
        proj_pooled_prompt = self.prompt_proj(pooled_prompt)  # (bs, d_proj)
        proj_hs = self.hs_proj(hs)  # (num_layer, bs, num_query, d_proj)

        # finally, get dot-product scores of shape (num_layer, bs, num_query, 1)
        scores = torch.matmul(proj_hs, proj_pooled_prompt.unsqueeze(-1))
        scores *= self.scale

        # clamp scores to a max value to avoid numerical issues in loss or matcher
        if self.clamp_logits:
            scores.clamp_(min=-self.clamp_max_val, max=self.clamp_max_val)

        return scores


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, Tensor] = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class TransformerWrapper(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        d_model: int,
        two_stage_type="none",  # ["none"] only for now
        pos_enc_at_input_dec=True,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.num_queries = decoder.num_queries if decoder is not None else None
        self.pos_enc_at_input_dec = pos_enc_at_input_dec

        # for two stage
        assert two_stage_type in ["none"], "unknown param {} of two_stage_type".format(
            two_stage_type
        )
        self.two_stage_type = two_stage_type

        self._reset_parameters()
        self.d_model = d_model

    def _reset_parameters(self):
        for n, p in self.named_parameters():
            if p.dim() > 1:
                if "box_embed" not in n and "query_embed" not in n and "reference_points" not in n:
                    nn.init.xavier_uniform_(p)


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float = 0.0,
        residual: bool = False,
        out_norm: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # whether to add the output as a residual connection to the input
        if residual and input_dim != output_dim:
            raise ValueError("residual is only supported if input_dim == output_dim")
        self.residual = residual
        # whether to apply a normalization layer to the output
        assert isinstance(out_norm, nn.Module) or out_norm is None
        self.out_norm = out_norm or nn.Identity()

    def forward(self, x):
        orig_x = x
        for i, layer in enumerate(self.layers):
            x = self.drop(F.relu(layer(x))) if i < self.num_layers - 1 else layer(x)
        if self.residual:
            x = x + orig_x
        x = self.out_norm(x)
        return x


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def get_clones_seq(module, N):
    return nn.Sequential(*[copy.deepcopy(module) for i in range(N)])


def get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def get_activation_module(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return nn.ReLU
    if activation == "gelu":
        return nn.GELU
    if activation == "glu":
        return nn.GLU
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def get_valid_ratio(mask):
    _, H, W = mask.shape
    valid_H = torch.sum(~mask[:, :, 0], 1)
    valid_W = torch.sum(~mask[:, 0, :], 1)
    valid_ratio_h = valid_H.float() / H
    valid_ratio_w = valid_W.float() / W
    valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
    return valid_ratio


def gen_sineembed_for_position(pos_tensor, num_feats=256):
    assert num_feats % 2 == 0
    num_feats = num_feats // 2
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(num_feats, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (torch.div(dim_t, 2, rounding_mode="floor")) / num_feats)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos


class SAM3Output(list):
    """
    A class representing the output of a SAM3 model.
    It provides an iterable interface that supports different iteration modes, including iterating over all steps per stage,
    last step per stage, and flattened output.
    Attributes:
        output: The output of the SAM3 model, represented as a list of lists.
        iter_mode: The current iteration mode.
    Example:
        >>> output = [[1, 2], [3, 4], [5, 6]]
        >>> sam3_output = SAM3Output(output)
        >>> for step in sam3_output:
        ...     print(step)
        [1, 2]
        [3, 4]
        [5, 6]
        >>> with SAM3Output.iteration_mode(SAM3Output.IterMode.LAST_STEP_PER_STAGE) as sam3_last_step_out:
        ...     for step in sam3_last_step_out:
        ...         print(step)
        [2]
        [4]
        [6]
        >>> with SAM3Output.iteration_mode(SAM3Output.IterMode.FLATTENED) as sam3_flattened_out:
        ...     for step in sam3_flattened_out:
        ...         print(step)
        1
        2
        3
        4
        5
        6
    """

    class IterMode(Enum):
        # Defines the type of iterator over ouptuts.
        ALL_STEPS_PER_STAGE = auto()
        LAST_STEP_PER_STAGE = auto()
        FLATTENED = auto()  # Returns each interactivity step as if it is a separate stage (this is used in SAM3Image model)

    def __init__(
        self,
        output: List[List[Dict]] = None,
        iter_mode: IterMode = IterMode.ALL_STEPS_PER_STAGE,
        loss_stages: Optional[List[int]] = None,
    ):
        if output is not None:
            assert isinstance(output, list) and len(output) > 0 and isinstance(output[0], list), (
                "Expected output to be a list of lists"
            )
            self.output = output
        else:
            self.output = []
        assert isinstance(iter_mode, SAM3Output.IterMode), (
            f"iter_mode shoulf be of enum type 'SAM3Output.IterMode'. Got {type(iter_mode)}"
        )

        self.iter_mode = iter_mode
        # We create a weak reference to self to be used in the lambda functions.
        # This is to avoid cyclic references and let SAM3Output be garabge collected.
        self_ref = weakref.ref(self)
        self._mode2iter = {
            SAM3Output.IterMode.ALL_STEPS_PER_STAGE: lambda: iter(self_ref().output),
            SAM3Output.IterMode.LAST_STEP_PER_STAGE: lambda: (
                inner_list[-1] for inner_list in self_ref().output
            ),
            SAM3Output.IterMode.FLATTENED: lambda: (
                element for inner_list in self_ref().output for element in inner_list
            ),
        }
        self.loss_stages = loss_stages

    @override
    def __iter__(self) -> Iterator:
        return self._mode2iter[self.iter_mode]()

    def __getitem__(self, index):
        """
        Returns the item at the specified index.
        Args:
            index (int): The index of the item to return.
        Returns:
            list or element: The item at the specified index.
        """
        assert isinstance(index, int), f"index should be an integer. Got {type(index)}"
        if self.iter_mode == SAM3Output.IterMode.ALL_STEPS_PER_STAGE:
            return self.output[index]
        elif self.iter_mode == SAM3Output.IterMode.LAST_STEP_PER_STAGE:
            return self.output[index][-1]
        elif self.iter_mode == SAM3Output.IterMode.FLATTENED:
            if index == -1:
                return self.self.output[-1][-1]
            else:
                flattened_output = sum(self.output, [])
                return flattened_output[index]

    class _IterationMode(AbstractContextManager):
        """
        A context manager that temporarily changes the iteration mode of a SAM3Output object.
        This class is used internally by the SAM3Output.iteration_mode method.
        """

        def __init__(self, model_output: "SAM3Output", iter_mode: "SAM3Output.IterMode"):
            self._model_output = model_output
            self._orig_iter_mode = model_output.iter_mode
            self._new_iter_mode = iter_mode

        @override
        def __enter__(self) -> "SAM3Output":
            self._model_output.iter_mode = self._new_iter_mode
            return self._model_output

        @override
        def __exit__(self, exc_type, exc_value, traceback):
            self._model_output.iter_mode = self._orig_iter_mode
            return super().__exit__(exc_type, exc_value, traceback)

    @staticmethod
    def iteration_mode(model_output: "SAM3Output", iter_mode: IterMode) -> _IterationMode:
        """
        Returns a context manager that allows you to temporarily change the iteration mode of the SAM3Output object.
        Args:
            model_output: The SAM3Output object.
            iter_mode: The new iteration mode.
        Returns:
            SAM3Output._IterationMode: A context manager that changes the iteration mode of the SAM3Output object.
        """
        return SAM3Output._IterationMode(model_output=model_output, iter_mode=iter_mode)

    def append(self, item: list):
        assert isinstance(item, list), f"Only list items are supported. Got {type(item)}"
        self.output.append(item)

    def __repr__(self):
        return self.output.__repr__()

    def __len__(self):
        if self.iter_mode in [
            SAM3Output.IterMode.ALL_STEPS_PER_STAGE,
            SAM3Output.IterMode.LAST_STEP_PER_STAGE,
        ]:
            return len(self.output)
        elif self.iter_mode == SAM3Output.IterMode.FLATTENED:
            flattened_output = sum(self.output, [])
            return len(flattened_output)
