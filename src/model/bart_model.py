# built-in
import copy
from dataclasses import dataclass
import math
import random
import warnings
from typing import Optional,Tuple,Dict,Any,List,Union,Callable
import os
# torch
import torch
from torch.nn import CrossEntropyLoss
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.utils.checkpoint
import numpy as np
# transformers
from transformers.trainer import Trainer
from transformers.trainer_callback import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.activations import ACT2FN
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput, Seq2SeqModelOutput, \
    BaseModelOutputWithPastAndCrossAttentions
from transformers.models.bart.modeling_bart import (
    shift_tokens_right,
    BartConfig,
    #BartEncoder,
    BartEncoderLayer,
    BartPretrainedModel,
    _expand_mask, _make_causal_mask,
    BartLearnedPositionalEmbedding, 
    BartAttention,
    #BartDecoder,
    BartDecoderLayer,
)
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
    ModelOutput,
)
from transformers.modeling_utils import unwrap_model,get_parameter_dtype
# own
from .graphtransformer import GraphTransformer,RGCN

class BartDecoder(BartPretrainedModel):
    """
    Transformer 解码器，由 *config.decoder_layers* 12个层组成。每层是一个 :class:`BartDecoderLayer`

    Args:
        config: BartConfig 类型的配置对象，包含解码器的各种参数设置。
        embed_tokens (nn.Embedding): 输出嵌入层，可选参数。
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = config.dropout   # 从配置中获取丢弃率，用于在训练时随机丢弃部分神经元，防止过拟合
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id  # 获取填充标记的索引，在处理输入序列时用于标识填充部分1
        self.max_target_positions = config.max_position_embeddings  # 获取最大的目标序列位置数，用于位置嵌入的设置
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        if embed_tokens is not None:    # 若传入了嵌入层，则使用传入的嵌入层
            self.embed_tokens = embed_tokens
        else:   # 若未传入嵌入层，根据配置创建一个新的嵌入层,该嵌入层将词索引转换为词向量，输入维度为词汇表大小，输出维度为模型的隐藏层维度
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)
        # 创建位置嵌入层，使用 BartLearnedPositionalEmbedding 类,为输入序列的每个位置学习一个嵌入向量，捕捉序列的顺序信息
        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        self.layers = nn.ModuleList([BartDecoderLayer(config) for _ in range(config.decoder_layers)])   # 对嵌入向量进行归一化处理，加速模型的训练过程
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            ).to(self.device)

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        r"""
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using :class:`~transformers.BartTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.

                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            encoder_hidden_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, encoder_sequence_length, hidden_size)`, `optional`):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, encoder_sequence_length)`, `optional`):
                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
                selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            head_mask (:obj:`torch.Tensor` of shape :obj:`(decoder_layers, decoder_attention_heads)`, `optional`):
                Mask to nullify selected heads of the attention modules. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            cross_attn_head_mask (:obj:`torch.Tensor` of shape :obj:`(decoder_layers, decoder_attention_heads)`, `optional`):
                Mask to nullify selected heads of the cross-attention modules in the decoder to avoid performing
                cross-attention on hidden heads. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
                Tuple of :obj:`tuple(torch.FloatTensor)` of length :obj:`config.n_layers`, with each tuple having 2
                tensors of shape :obj:`(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional
                tensors of shape :obj:`(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see :obj:`past_key_values` input) to speed up sequential
                decoding.

                If :obj:`past_key_values` are used, the user can optionally input only the last
                :obj:`decoder_input_ids` (those that don't have their past key value states given to this model) of
                shape :obj:`(batch_size, 1)` instead of all :obj:`decoder_input_ids`` of shape :obj:`(batch_size,
                sequence_length)`.
            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # embed positions
        positions = self.embed_positions(input_shape, past_key_values_length)

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                assert attn_mask.size()[0] == (
                    len(self.layers)
                ), f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                if use_cache:
                    '''logger.warning(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )'''
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,
                )
            else:

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )

class BartEncoder(BartPretrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    :class:`BartEncoderLayer`.

    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )
        self.layers = nn.ModuleList([BartEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(embed_dim)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        r"""
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using :class:`~transformers.BartTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.

                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            head_mask (:obj:`torch.Tensor` of shape :obj:`(encoder_layers, encoder_attention_heads)`, `optional`):
                Mask to nullify selected heads of the attention modules. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        embed_pos = self.embed_positions(input_shape)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if getattr(self.config, "gradient_checkpointing", False) and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


# 多头注意力机制是 Transformer 架构的核心组件之一，它允许模型在处理输入序列时，能够同时关注序列的不同部分，从而更好地捕捉序列中的依赖关系和语义信息。该类可以用于编码器或解码器中，根据 is_decoder 参数来区分不同的使用场景。
class BartAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads})."
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)      # 定义线性投影层，将多头注意力的输出投影回原始的嵌入维度

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        # [bs,seq_len,hid_dim]
        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling # [bs,seq_len,emb_dim]
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz) #[bs,num_heads,seq_len,head_dim]
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = F.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value
#   add a new speaker attention layer in the encoder and deocder just like  that in the "Hierarchical Learning for Generation with Long Source Sequences"
@dataclass
class DualBaseModelOutput(ModelOutput):
    """
    This is DualBaseModelOutput for dual encoder outputs: low_encoder and high_encoder
    The original member for BaseModelOutput is still the same
    1.last_hidden_state
    2.hidden_states
    3.attentions
    We add additional members:
    1.speaker_hidden_states
    2.speaker_attentions
    3.speaker_attention_mask(for generation)
    """
    low_encoder_last_hidden_state: torch.FloatTensor = None
    low_encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    low_encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    low_encoder_attention_mask: torch.LongTensor = None

    high_encoder_last_hidden_state: torch.FloatTensor = None
    high_encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    high_encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    high_encoder_attention_mask:  torch.LongTensor = None

class BartLearnedSpeakerEmbedding(nn.Module):
    def __init__(self,num_embeddings,embedding_dim):
        super().__init__()
        self.speaker_embedding = nn.Embedding(num_embeddings,embedding_dim)
    def forward(self,speaker_type_ids):
        # speaker_type_id : [bs,seq_len-2]
        out = self.speaker_embedding(speaker_type_ids)

        return out


# 构建一个分层的 Transformer 编码器。该编码器由多个自注意力层（BartEncoderLayer）组成
class HierarchicalEncoder(BartPretrainedModel):
    """
    Transformer 编码器，由 *config.encoder_layers* 12个自注意力层组成。
    每层是一个 :class:`BartEncoderLayer`。

    Args:
        config: BartConfig 类型的配置对象，包含模型的各种参数设置。
        embed_tokens (nn.Embedding): 输出嵌入层，可选参数。
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)

        self.config = config
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model  # 获取模型的隐藏层维度
        self.padding_idx = config.pad_token_id      # 获取填充标记的索引，用于处理输入序列中的填充部分
        self.max_source_positions = config.max_position_embeddings  # 获取最大的源序列位置数，用于位置嵌入
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0  # 计算嵌入层的缩放因子，如果配置中要求缩放嵌入，则为嵌入维度的平方根，否则为 1.0
        # 如果传入了嵌入层，则使用传入的嵌入层
        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)
        # 创建位置嵌入层，使用 BartLearnedPositionalEmbedding 类，该层用于为输入序列的每个位置学习一个嵌入向量
        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )
        self.layers = nn.ModuleList([BartEncoderLayer(config) for _ in range(config.encoder_layers)])# 创建编码器层列表，每个编码器层是一个 BartEncoderLayer 实例，编码器层的数量由 config.encoder_layers 12决定
        self.layernorm_embedding = nn.LayerNorm(embed_dim)          # 创建嵌入层的层归一化层，用于对嵌入向量进行归一化处理

        graphtransformer_config = config
        if config.ablation_type != 'rgcn':      # 根据消融实验类型选择不同的图模型
            self.graphtransformer =GraphTransformer(graphtransformer_config)
        else:
            self.graphtransformer = RGCN(graphtransformer_config)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        """
        input_ids:
            torch.Tensor [bs,seq_len]
        attention_mask: 
            torch.Tensor [bs,seq_len]
        gt_input_ids: 
            torch.Tensor [bs*num_utt,seq_len]
        gt_attention_mask: 
            torch.Tensor [bs*num_utt,seq_len]
        """
        gt_input_ids = kwargs.get('gt_input_ids',None)
        gt_attention_mask = kwargs.get('gt_attention_mask',None)
        num_utt_ls = kwargs.get('num_utt_ls',None)

        bs = input_ids.shape[0]
        max_utt_num_in_a_batch = max(num_utt_ls) if num_utt_ls else max([x.shape[0] for x in gt_input_ids])      # 计算批次中最大的话语数量
        #print("num_utt_ls:",[x.shape[0] for x in gt_input_ids],"max_seq_len:",max([x.shape[1] for x in gt_input_ids]))
        device = gt_input_ids[0].device if isinstance(gt_input_ids,list) else gt_input_ids.device

        low_encoder_output_batched = []
        low_encoder_attention_mask_batched = []
        if isinstance(gt_input_ids,list):       # 处理 gt_input_ids 为列表的情况
            for _input_ids,_attention_mask in zip(gt_input_ids,gt_attention_mask):
                
                num_utt_in_a_conv  = _input_ids.shape[0]
                diff = max_utt_num_in_a_batch - num_utt_in_a_conv
                # [num_utt,seq_len,hid_dim]
                low_encoder_last_hidden_state = self.get_low_encoder_output(_input_ids,_attention_mask,with_grad=True).last_hidden_state
                
                # [num_utt,hid_dim]
                low_encoder_last_hidden_state = self.extract_from_utt(low_encoder_last_hidden_state)

                # pad input_for_high_encoder to [max_utt_num,hid_dim]
                index_ls = list(range(num_utt_in_a_conv))+[0]*(diff) # 0 is randomly picked since it will be masked
                indices = torch.tensor(index_ls).to(device)
                low_encoder_last_hidden_state = low_encoder_last_hidden_state.index_select(0,indices)
                low_encoder_output_batched.append(low_encoder_last_hidden_state)

                # generate_new_attention_mask
                new_attention_mask = [1]*num_utt_in_a_conv + [0]* diff
                low_encoder_attention_mask_batched.append(new_attention_mask)
            
            # [bs,num_utt,hid_dim]
            low_encoder_output_batched = torch.stack(low_encoder_output_batched) 

            # [bs,num_utt]
            low_encoder_attention_mask_batched = torch.tensor(low_encoder_attention_mask_batched,dtype=torch.int32).to(device)
        elif isinstance(gt_input_ids,torch.Tensor):     # 处理 gt_input_ids 为张量的情况
            # 调用低层级编码器对输入进行编码，得到最后一层的隐藏状态
            # low_encoder_last_hidden_state: [1*5,15,1024]
            low_encoder_output_batched  = self.get_low_encoder_output(gt_input_ids,gt_attention_mask,with_grad=True).last_hidden_state

            d_model = low_encoder_output_batched.shape[-1]  # 获取隐藏状态的最后一个维度，即隐藏层维度1024
            low_encoder_output_batched = self.extract_from_utt(low_encoder_output_batched,gt_attention_mask=gt_attention_mask)  # 从低层级编码器的输出中提取特征，将形状变为 [bs*num_utt_per_batch, hid_dim]（5，1024）
            index_ls = []   # 用于存储索引的列表
            for num_utt in num_utt_ls:
                diff = max_utt_num_in_a_batch - num_utt
                ls = list(range(num_utt)) + diff * [num_utt-1]
                if index_ls: offset = index_ls[-1]+1
                else:offset = 0
                ls = [x+offset for x in ls]
                index_ls.extend(ls)         # 话语数量与批次中最大话语数量
            index_ls= torch.tensor(index_ls).to(device)
            low_encoder_output_batched = low_encoder_output_batched.index_select(0,index_ls)    # 使用 index_select 方法根据 index_ls 对 low_encoder_output_batched 进行索引选择，
            low_encoder_output_batched = low_encoder_output_batched.view(bs,max_utt_num_in_a_batch,d_model) # （1，5，1024）实现填充操作，使每个样本的话语数量达到 max_utt_num_in_a_batch
            high_encoder_attention_mask_batched = torch.tensor(
                [[1]*x + [0]*(max_utt_num_in_a_batch-x) for x in num_utt_ls]
            ).to(device)    # 将num_utt_ls列表组合成一个二维列表，并转换为 torch.Tensor，移动到指定设备 device 上，得到形状为 [1, 5] 的注意力掩码
        # 加权计算部分（假设在 HierarchicalEncoder 的 forward 方法中）
        weighted_output = low_encoder_output_batched.clone()
        for b in range(bs):
            num_utt = num_utt_ls[b]
            if num_utt == 0:
                continue
            sens_emb = low_encoder_output_batched[b, :num_utt, :]  # [num_utt, hid_dim]

            # 计算GLC分数
            gl_scores, _, _ = get_global_local_centrality_score(sens_emb)
            gl_scores = np.array(gl_scores)

            # 单次计算GLC分数（仅全局中心性）
            #gl_scores, _, _ = get_global_local_centrality_score1(sens_emb)
            #gl_scores = np.array(gl_scores)
            # 归一化权重
            gl_scores = gl_scores / (np.linalg.norm(gl_scores) + 1e-10)
            gl_scores_tensor = torch.tensor(gl_scores, dtype=torch.float32, device=device)
            new_weights = gl_scores_tensor.unsqueeze(-1).expand(num_utt, d_model)

            lam = 0.5 if num_utt > 15 else 0.3  # 长对话用0.5，短对话用0.3
            weighted_output[b, :num_utt, :] = lam * (sens_emb * new_weights) + (1 - lam) * sens_emb
        # 将低层级编码器的输出输入到图模型中进行处理
        graphtransformer_output = self.graphtransformer(
            inputs_embeds = weighted_output,             # 低层级编码器的输出
            attention_mask = high_encoder_attention_mask_batched,   # tensor([[1, 1, 1, 1, 1]], device='cuda:0')
            **kwargs,
        )

        # 对整体输入进行低层级编码
        low_encoder_output_batched = self.get_low_encoder_output(
        input_ids,attention_mask,with_grad=True
        ).last_hidden_state
        low_encoder_attention_mask_batched = attention_mask
        # 返回包含低层级和高层级编码结果的对象
        return DualBaseModelOutput(
            low_encoder_last_hidden_state = low_encoder_output_batched,     # (1,64,1024)
            low_encoder_attention_mask = low_encoder_attention_mask_batched,

            high_encoder_last_hidden_state=graphtransformer_output.last_hidden_state,    # (1,5,1024)
            high_encoder_attention_mask = high_encoder_attention_mask_batched,
        )

    def extract_from_utt(self,low_encoder_last_hidden_state,gt_attention_mask=None):
        #low_encoder_last_hidden_state: [bs*num_utt,seq_len,d_model]
        #gt_attention_mask: [bs*num_utt,seq_len]
        if self.config.utt_pooling == 'extract':
            return low_encoder_last_hidden_state[:,0,:] # extract <s> 
        elif self.config.utt_pooling == 'average':
            return torch.sum(low_encoder_last_hidden_state * gt_attention_mask.unsqueeze(-1),dim=1)
    
    def get_low_encoder_output(self,input_ids,attention_mask,with_grad=False):

        input_shape = input_ids.size()   # 获取输入张量 input_ids 的形状，通常返回一个元组，例如 (5, 15)
        input_ids = input_ids.view(-1, input_shape[-1]) # 对 input_ids 进行形状调整，将其变为二维张量，形状为 [-1, input_shape[-1]]，这里 -1 表示该维度的大小由其他维度自动推断(5,15)
        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale # 将输入的词 ID 转换为嵌入向量，并进行缩放

        embed_pos = self.embed_positions(input_shape)   # 调用位置嵌入层 self.embed_positions，传入输入的形状 input_shape，得到输入序列每个位置对应的位置嵌入向量

        hidden_states = inputs_embeds + embed_pos   # 将输入的嵌入向量和位置嵌入向量相加，为输入序列添加位置信息，得到包含位置信息的隐藏状态
        hidden_states = self.layernorm_embedding(hidden_states)     # 对包含位置信息的隐藏状态进行层归一化处理
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)    # 对隐藏状态应用丢弃（dropout）操作，以一定的概率 p（self.dropout）随机将部分元素置为 0，

        # 如果传入了注意力掩码 attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)  # 调用 _expand_mask 函数将注意力掩码进行扩展
        
        dropout_probability = random.uniform(0, 1)  # 生成一个 0 到 1 之间的随机浮点数，用于 LayerDrop 机制
        if self.training and (dropout_probability < self.layerdrop):  # 判断是否处于训练模式，并且随机数小于层丢弃率 self.layerdrop
            layer_outputs = (None, None)
        else:
            for idx, encoder_layer in enumerate(self.layers):   # 遍历编码器的每一层
                if with_grad:   # 如果需要计算梯度
                    layer_outputs = encoder_layer(   # 调用当前编码器层 encoder_layer 的前向传播方法，传入隐藏状态、注意力掩码等参数，
                                hidden_states,       # layer_head_mask 设为 None 表示不使用头掩码，output_attentions 设为 False 表示不输出注意力分数
                                attention_mask,
                                layer_head_mask=None,
                                output_attentions=False,
                    )
                else:
                    with torch.no_grad():
                        layer_outputs = encoder_layer(
                                hidden_states,
                                attention_mask,
                                layer_head_mask=None,
                                output_attentions=False,
                    )

                hidden_states = layer_outputs[0]    # 从当前层的输出中获取更新后的隐藏状态，用于下一层的输入

        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=None, attentions=None,
        )    # 这里 hidden_states 和 attentions 都设为 None 表示不返回各层隐藏状态和注意力分数
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention mask from [bsz, seq_len] to [bsz, 1, tgt_seq_len, src_seq_len].
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)
def get_centrality_score1(embs):
    matrix = torch.matmul(embs, embs.T)
    raw_scores = torch.sum(matrix, dim=1)
    normalized_scores = F.normalize(raw_scores, p=2, dim=0)
    return normalized_scores

def get_global_local_centrality_score1(sens_emb):
    torch.manual_seed(42)
    gl_scores = get_centrality_score1(sens_emb).tolist()  # 直接计算全局中心性
    return gl_scores, None, None
# GLC相关函数
def kmeans(x, ncluster, niter=10):
    N, D = x.size()
    c = x[torch.randperm(N)[:ncluster]]
    for i in range(niter):
        a = ((x[:, None, :] - c[None, :, :])**2).sum(-1).argmin(1)
        c = torch.stack([x[a == k].mean(0) for k in range(ncluster)])
        nanix = torch.any(torch.isnan(c), dim=1)
        ndead = nanix.sum().item()
        c[nanix] = x[torch.randperm(N)[:ndead]]
    dist = ((x[:, None, :] - c[None, :, :])**2).sum(-1)
    return c, dist

def get_centrality_score(embs):
    """
    计算话语嵌入的中心性得分。
    参数:
    embs (torch.Tensor): 话语嵌入张量，形状为 [num_utt, hid_dim]
    返回:
    torch.Tensor: 每个话语的中心性得分，形状为 [num_utt]
    """
    matrix = torch.matmul(embs, embs.T)  # 形状: [num_utt, num_utt]
    raw_scores = torch.sum(matrix, dim=1)  # 形状: [num_utt]
    normalized_scores = F.normalize(raw_scores, p=2, dim=0)  # 形状: [num_utt]
    return normalized_scores

def get_cluster_id(dist_matrix):
    return dist_matrix.argmin(1)

def get_index_lst(c, cluster_id_lst):
    return [index for index, idx in enumerate(cluster_id_lst) if c == idx]

def get_global_score(clusters):
    return get_centrality_score(clusters)

def get_local_score(cluster_id_lst, embs):
    local_scores = torch.zeros(len(cluster_id_lst), device=embs.device)
    n = torch.max(cluster_id_lst)
    for c in range(n + 1):
        new_embs = embs[(c == cluster_id_lst)]
        index_lst = get_index_lst(c, cluster_id_lst)
        if len(new_embs) == 0:
            continue
        elif len(new_embs) == 1:
            local_scores[index_lst[0]] = 1.0
        else:
            cen_scores = get_centrality_score(new_embs)
            for index, score in zip(index_lst, cen_scores):
                local_scores[index] = score
    return local_scores

def get_global_local_centrality_score(sens_emb):
    """
    计算全局和局部中心性得分，短对话直接用全局中心性，长对话用聚类。
    参数:
    sens_emb (torch.Tensor): 话语嵌入张量，形状为 [num_utt, hid_dim]
    返回:
    list: 综合中心性得分列表
    None: 占位符（兼容原始接口）
    None: 占位符（兼容原始接口）
    """
    torch.manual_seed(42)  # 固定随机种子
    if len(sens_emb) < 10:  # 短对话直接计算全局中心性
        gl_scores = get_centrality_score(sens_emb).tolist()
        return gl_scores, None, None
    else:  # 长对话使用聚类
        ncluster = min(len(sens_emb) // 5, 3)  # 动态调整聚类数
        clusters, dist = kmeans(sens_emb, ncluster, niter=20)
        cluster_id_lst = get_cluster_id(dist)
        global_scores = get_global_score(clusters)
        local_scores = get_local_score(cluster_id_lst, sens_emb)
        gl_scores = [l.item() * global_scores[index].item() for l, index in zip(local_scores, cluster_id_lst)]
        return gl_scores, global_scores, local_scores

# 主要的功能是构建一个带有双交叉注意力机制的 BART 解码器
class BartDecoderWithDualCrossAttention(BartDecoder):
    
    #  replace BartDecoderLayer with BartDecoderLayerWithSpeakerAttention
    def __init__(self,config:BartConfig,embed_tokens:Optional[nn.Embedding] = None):
        super().__init__(config,embed_tokens=None)
        self.layers = nn.ModuleList([BartDecoderLayerWithDualCrossAttention(config) for _ in  range(config.decoder_layers)])
        # 创建解码器层列表，根据配置中的 decoder_layers 参数确定层数，创建对应数量的 BartDecoderLayerWithDualCrossAttention 实例
        # 每个 BartDecoderLayerWithDualCrossAttention 实例会处理目标序列的一部分特征，并通过双交叉注意力机制与输入序列进行交互
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,        
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,

        # added
        **kwargs
    ):
        low_encoder_last_hidden_state = kwargs['low_encoder_last_hidden_state']
        low_encoder_attention_mask = kwargs['low_encoder_attention_mask']

        high_encoder_last_hidden_state = kwargs['high_encoder_last_hidden_state']
        high_encoder_attention_mask = kwargs['high_encoder_attention_mask']

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # expand encoder attention mask
        if low_encoder_last_hidden_state is not None and low_encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            kwargs['low_encoder_attention_mask'] = _expand_mask(low_encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
        
        
        if high_encoder_last_hidden_state is not None and high_encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            kwargs['high_encoder_attention_mask'] = _expand_mask(high_encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
        
        # embed positions
        positions = self.embed_positions(input_shape, past_key_values_length)

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and low_encoder_last_hidden_state is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                assert attn_mask.size()[0] == (
                    len(self.layers)
                ), f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                # if use_cache:
                #     logger.warning(
                #         "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                #         "`use_cache=False`..."
                #     )
                #     use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    low_encoder_last_hidden_state,
                    low_encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,
                )
            else:

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    **kwargs,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                # if encoder_hidden_states is not None:
                #     all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


# 每个 BartDecoderLayerWithDualCrossAttention 实例会处理目标序列的一部分特征，并通过双交叉注意力机制与输入序列进行交互
class BartDecoderLayerWithDualCrossAttention(BartDecoderLayer):
    def __init__(self,config:BartConfig):
        super().__init__(config)
        
        # 额外的多头注意力层，用于处理说话者注意力或其他额外信息,该注意力层会与原有的交叉注意力层协同工作，实现双交叉注意力机制
        self.second_crossattn = BartAttention(
            embed_dim=self.embed_dim,       # 嵌入维度，与当前解码器层的嵌入维度保持一致1024
            num_heads = config.gt_decoder_attention_heads,  # 注意力头的数量，从配置中获取8
            dropout = config.gt_attention_dropout,
            is_decoder = True,      # 标记该注意力层用于解码器
        )
        self.second_crossattn_layer_norm = nn.LayerNorm(self.embed_dim)     # 对第二个交叉注意力层的输出进行层归一化操作
        if config.rezero != -1:             # 根据配置中的 rezero 参数决定是否使用可学习的残差权重
            self.resweight = nn.Parameter(torch.Tensor([config.rezero]))
        else:
            self.resweight = 1
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        # encoder_hidden_states: Optional[torch.Tensor] = None,
        # encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,

        #
        **kwargs
    ):

        low_encoder_last_hidden_state = kwargs['low_encoder_last_hidden_state']
        low_encoder_attention_mask = kwargs['low_encoder_attention_mask']

        high_encoder_last_hidden_state = kwargs['high_encoder_last_hidden_state']
        high_encoder_attention_mask = kwargs['high_encoder_attention_mask']
        residual = hidden_states

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        # first-cross-attention
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if low_encoder_last_hidden_state is not None:
            residual = hidden_states

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=low_encoder_last_hidden_state,
                attention_mask=low_encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
            
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # second_crossattn
        #high_encoder_last_hidden_state = None
        if high_encoder_last_hidden_state is not None:
            residual = hidden_states
            hidden_states,*_= self.second_crossattn(
                hidden_states = hidden_states,
                key_value_states = high_encoder_last_hidden_state,
                attention_mask = high_encoder_attention_mask,
                output_attentions = output_attentions,
            )
            hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
            #print("rezero_factor:",self.resweight)
            hidden_states = residual + hidden_states* self.resweight
            hidden_states = self.second_crossattn_layer_norm(hidden_states)
        
        
        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


# 该类主要负责初始化模型的各个组件，包括共享嵌入层、编码器和解码器，并对模型的权重进行初始化。
class BartModel(BartPretrainedModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = HierarchicalEncoder(config, self.shared)
        self.decoder = BartDecoderWithDualCrossAttention(config, self.shared)

        self.init_weights()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        # if isinstance(encoder_outputs,BaseModelOutput):
        #     encoder_hidden_states = encoder_outputs.last_hidden_state
        # elif isinstance(encoder_outputs,DualBaseModelOutput):
        #     encoder_hidden_states = encoder_outputs.low_encoder_last_hidden_state
        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        kwargs.update(encoder_outputs)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            # encoder_hidden_states=encoder_hidden_states,
            # encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            #encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            #encoder_hidden_states=encoder_outputs.hidden_states,
            #encoder_attentions=encoder_outputs.attentions,
        )


# 核心功能是构建 BART 模型的基础结构。它能够依据配置信息，灵活选择不同类型的编码器和解码器，进而组成不同结构的 BART 模型
class BaseBart(BartPretrainedModel):
    """
    BaseBart 类继承自 BartPretrainedModel，用于构建 BART 模型的基础结构。
    它可以根据配置选择不同的编码器和解码器，可作为 BartForConditionalGeneration 的基础模型。
    同时，这里也是融合 SpeakerGraph（可能是一种特定的图结构）的地方，但此部分与模型输出无关。
    """
    def __init__(self,config:BartConfig):
        super().__init__(config)
        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx) # 创建一个共享的嵌入层，用于将输入的词索引转换为词向量，嵌入层的输入维度为词汇表大小50265，输出维度为模型的隐藏层维度1024
        # 根据配置中的模型类型选择不同的编码器和解码器
        if config.model_type == 'baseline':
            self.encoder = BartEncoder(config, self.shared)
            self.decoder = BartDecoder(config, self.shared)
        elif config.model_type == 'graphtransformer':
            self.encoder = HierarchicalEncoder(config,self.shared)                   # 这种编码器具有分层结构，以适应图Transformer的需求
            self.decoder = BartDecoderWithDualCrossAttention(config, self.shared)    # 这种解码器具有双重交叉注意力机制
        else:
            print("请指定模型类型")
        self.init_weights()
    
    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        ## additional
        **kwargs
    ):
        # 与其他模型不同，Bart 模型会在没有提供 decoder_input_ids 时，自动从 input_ids 创建 decoder_input_ids
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )
    
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (        # 如果没有提供 output_hidden_states 参数，则使用配置中的 output_hidden_states 设置
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache   # 如果没有提供 use_cache 参数，则使用配置中的 use_cache 设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict   # 如果没有提供 return_dict 参数，则使用配置中的 use_return_dict 设置
        # 如果没有提供编码器的输出结果
        if encoder_outputs is None:
            encoder_outputs = self.encoder(     # 调用编码器进行编码操作
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs
            )
        
        kwargs.update(encoder_outputs)      # 将编码器的输出结果更新到 kwargs 中，方便后续使用

        if isinstance(encoder_outputs,BaseModelOutput):
            encoder_hidden_states = encoder_outputs.last_hidden_state
            encoder_attention_mask = attention_mask
        elif isinstance(encoder_outputs,DualBaseModelOutput):
            encoder_hidden_states = None,
            encoder_attention_mask = None
        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            #encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            #encoder_hidden_states=encoder_outputs.hidden_states,
            #encoder_attentions=speaker_attentions,
        )


# 用于构建一个基于 BART 架构的模型,初始化基础模型 BaseBart，它是自定义的基础 BART 模型类，实现了具体的模型结构。
class BART(BartPretrainedModel):
    base_model_prefix = "model"     # 定义基础模型的前缀，在保存和加载模型时会用到
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head\.weight"]      # r"final_logits_bias" 表示可以忽略名为 "final_logits_bias" 的键

    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.model = BaseBart(config)   # 创建 BaseBart 模型实例，传入配置对象,BaseBart 是自定义的基础 BART 模型类，用于实现具体的模型结构
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))   # 注册一个缓冲区 "final_logits_bias"，用于存储最终 logits 的偏置，初始化为全零张量，形状为 (1, self.model.shared.num_embeddings)
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)      # 定义语言模型头，是一个线性层，输入维度为1024，输出维度

        self.init_weights()     # 初始化模型的权重

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids=None,# 输入序列的词元 ID，通常是经过分词器处理后的输入文本对应的 ID 序列
        attention_mask=None,
        decoder_input_ids=None,# 解码器的输入词元 ID，用于生成目标序列
        decoder_attention_mask=None,# 解码器的注意力掩码
        head_mask=None,# 编码器的头掩码，用于控制编码器中多头注意力机制里各个头的开启或关闭
        decoder_head_mask=None,
        cross_attn_head_mask=None,# 交叉注意力的头掩码，用于控制编码器 - 解码器交叉注意力机制里各个头的开启或关闭
        encoder_outputs=None,# 编码器的输出结果，通常是编码器对输入序列编码后得到的隐藏状态
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,# 目标序列的标签，用于计算损失
        use_cache=None,
        output_attentions=True,# 是否输出注意力分数，默认为 True
        output_hidden_states=None,# 是否输出隐藏状态，默认为 None
        return_dict=None,# 是否以字典形式返回输出结果，默认为 None

        # additional
        **kwargs

    ):
        # 如果 return_dict 未提供，则使用配置中的 use_return_dict 参数
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 如果提供了标签
        if labels is not None:
            # 若解码器输入 ID 未提供
            if decoder_input_ids is None:
                # 调用 shift_tokens_right 函数，将标签进行右移操作，生成解码器输入 ID。 右移操作是为了在解码器输入时，将目标序列的每个词元作为下一个时间步的输入
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
        # 调用模型的前向传播方法，传入各种输入参数
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias    # 通过语言模型头（lm_head）对解码器的输出进行线性变换，得到语言模型的对数概率（logits）并加上最终的对数偏置

        masked_lm_loss = None       # 初始化掩码语言模型损失为 None
        if labels is not None:
            loss_fct = CrossEntropyLoss()   # 定义交叉熵损失函数
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))  # 将语言模型的对数概率和标签进行展平处理后，传入损失函数计算损失

        if not return_dict:     # 如果不使用字典形式返回输出结果
            output = (lm_logits,) + outputs[1:]     # 将语言模型的对数概率和模型的其他输出结果拼接成一个元组
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output   # 如果存在掩码语言模型损失，将损失添加到输出元组的开头
        # 返回一个 Seq2SeqLMOutput 对象，包含损失、对数概率、过去的键值对、解码器隐藏状态等信息
        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]
        
        if kwargs.get('labels',None) != None:
            kwargs.pop('labels')
    
        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "use_cache": use_cache, # change this to avoid caching (presumably for debugging)
            **kwargs  
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past
    # for beam_search 
    
    @staticmethod
    def _expand_inputs_for_generation(
        input_ids,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        attention_mask: torch.LongTensor = None,
        encoder_outputs: DualBaseModelOutput = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(0, expanded_return_idx)

        if attention_mask is not None:
            if isinstance(attention_mask,list):
                model_kwargs["attention_mask"] = [m.index_select(0, expanded_return_idx) for m in attention_mask]
            else:
                model_kwargs["attention_mask"] = attention_mask.index_select(0, expanded_return_idx)
        
        # if 'special_tokens_mask' in model_kwargs.keys() is not None:
        #    model_kwargs['special_tokens_mask'] = model_kwargs['special_tokens_mask'].index_select(0,expanded_return_idx)
           
        if is_encoder_decoder:
            assert encoder_outputs is not None
            device = encoder_outputs[0].device
            if "last_hidden_state" in  encoder_outputs.keys():
                encoder_outputs["last_hidden_state"] = encoder_outputs["last_hidden_state"].index_select(
                0, expanded_return_idx.to(device)
            )
            if "low_encoder_last_hidden_state" in encoder_outputs.keys():
                encoder_outputs["low_encoder_last_hidden_state"] = encoder_outputs["low_encoder_last_hidden_state"].index_select(
                0, expanded_return_idx.to(device)
            )
            if "high_encoder_last_hidden_state" in encoder_outputs.keys():
                encoder_outputs["high_encoder_last_hidden_state"] = encoder_outputs["high_encoder_last_hidden_state"].index_select(
                0, expanded_return_idx.to(device)
            )
            if "high_encoder_attention_mask" in encoder_outputs.keys():
                encoder_outputs["high_encoder_attention_mask"] = encoder_outputs["high_encoder_attention_mask"].index_select(
                0, expanded_return_idx.to(device)
            )
            if "low_encoder_attention_mask" in encoder_outputs.keys():
                encoder_outputs["low_encoder_attention_mask"] = encoder_outputs["low_encoder_attention_mask"].index_select(
                0, expanded_return_idx.to(device)
            )

            model_kwargs["encoder_outputs"] = encoder_outputs
        return input_ids, model_kwargs

    def estimate_tokens(self, input_dict: Dict[str, Union[torch.Tensor, Any]]) -> int:
        """
        Helper function to estimate the total number of tokens from the model inputs.

        Args:
            inputs (:obj:`dict`): The model inputs.

        Returns:
            :obj:`int`: The total number of tokens.
        """
        token_inputs = [tensor for key, tensor in input_dict.items() if "input" in key]
        if token_inputs:
            ret = 0
            for token_input in token_inputs:
                if isinstance(token_input,list) and isinstance(token_input[0],torch.Tensor):
                    ret += sum([_token_input.numel() for _token_input in token_input])
                elif isinstance(token_input,torch.Tensor):
                    ret += token_input.numel()
            return ret
        else:
            warnings.warn(
                "Could not estimate the number of tokens of the input, floating-point operations will not be computed"
            )
            return 0
    
    def _prepare_decoder_input_ids_for_generation(
        self, input_ids: torch.LongTensor, decoder_start_token_id: int = None, bos_token_id: int = None
    ) -> torch.LongTensor:
        decoder_start_token_id = self._get_decoder_start_token_id(decoder_start_token_id, bos_token_id)
        if isinstance(input_ids,list):
            bs = len(input_ids)
            device = input_ids[0].device
        else: 
            bs = input_ids.shape[0]
            device = input_ids.device
        decoder_input_ids = (
            torch.ones((bs, 1), dtype=torch.long, device=device) * decoder_start_token_id
        )
        return decoder_input_ids
    
    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        save_config: bool = False,
        state_dict: Optional[dict] = None,
        save_function: Callable = torch.save,
        push_to_hub: bool = False,
        **kwargs,
    ):
        WEIGHTS_NAME = "pytorch_model.bin"
        os.makedirs(save_directory, exist_ok=True)

        # Only save the model itself if we are using distributed training
        model_to_save = unwrap_model(self)

        # save the string version of dtype to the config, e.g. convert torch.float32 => "float32"
        # we currently don't use this setting automatically, but may start to use with v5
        dtype = get_parameter_dtype(model_to_save)
        model_to_save.config.torch_dtype = str(dtype).split(".")[1]

        # Attach architecture to the config
        model_to_save.config.architectures = [model_to_save.__class__.__name__]

        # Save the config
        if save_config:
            model_to_save.config.save_pretrained(save_directory)

        # Save the model
        if state_dict is None:
            state_dict = model_to_save.state_dict()

        # Handle the case where some state_dict keys shouldn't be saved
        if self._keys_to_ignore_on_save is not None:
            state_dict = {k: v for k, v in state_dict.items() if k not in self._keys_to_ignore_on_save}

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, WEIGHTS_NAME)
        save_function(state_dict, output_model_file)

        #logger.info(f"Model weights saved in {output_model_file}")

