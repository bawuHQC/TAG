from networkx.classes.function import selfloop_edges
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import dgl
import dgl.nn as dglnn
from transformers.models.marian.modeling_marian import \
    MarianSinusoidalPositionalEmbedding as SinusoidalPositionalEmbedding
from transformers.models.bart.modeling_bart import (
    shift_tokens_right,
    BartConfig,
    BartEncoder,
    BartEncoderLayer,
    BartPretrainedModel,
    _expand_mask, _make_causal_mask,
    BartLearnedPositionalEmbedding,
    BartAttention,
    BartDecoder,
    BartDecoderLayer,
)
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput, Seq2SeqModelOutput, \
    BaseModelOutputWithPastAndCrossAttentions
from src.utils.CONSTANT import DISCOURSE_RELATIONS

from transformers import BartConfig

name_2_activation_fn_mapping = {
    'tanh': F.tanh,
    'relu': F.relu,
}


# 配置图 Transformer 模型的各种参数
class GraphTransformerConfig(BartConfig):

    def __init__(
            self,
            backbone_model='../pretrained_model/bart_large',
            model_type='graphtransformer',
            # all_bart_base config
            gt_activation_dropout=0.1,
            gt_activation_function='gelu',
            gt_add_bias_logits=False,
            gt_add_final_layer_norm=False,
            gt_attention_dropout=0.1,
            gt_d_model=768,
            gt_decoder_attention_heads=12,
            gt_decoder_ffn_dim=3072,
            gt_decoder_layerdrop=0.0,
            gt_dropout=0.1,
            gt_encoder_attention_heads=12,
            gt_encoder_ffn_dim=3072,
            gt_encoder_layerdrop=0.0,
            gt_encoder_layers=6,
            gt_init_std=0.02,
            gt_is_encoder_decoder=True,
            gt_normalize_before=False,
            gt_normalize_embedding=True,
            gt_scale_embedding=False,
            conv_activation_fn='relu',
            num_beams=5,
            rezero=1,
            max_length=100,
            min_length=5,
            utt_pooling='average',
            gt_pos_embed='',
            **kwargs,
    ):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                setattr(self, k, v)

        pretrained_model_config = BartConfig.from_pretrained(backbone_model)
        for k, v in vars(pretrained_model_config).items():
            if not hasattr(self, k):
                setattr(self, k, v)
        self.gt_pos_embed = gt_pos_embed
        self.conv_activation_fn = conv_activation_fn
        self.utt_pooling = utt_pooling
        self.backbone_model = backbone_model
        self.model_type = model_type
        self.gt_activation_dropout = gt_activation_dropout
        self.gt_activation_function = gt_activation_function
        self.gt_add_bias_logits = gt_add_bias_logits
        self.gt_add_final_layer_norm = gt_add_final_layer_norm
        self.gt_attention_dropout = gt_attention_dropout
        self.gt_d_model = gt_d_model
        self.gt_decoder_attention_heads = gt_decoder_attention_heads
        self.gt_decoder_ffn_dim = gt_decoder_ffn_dim
        self.gt_decoder_layerdrop = gt_decoder_layerdrop

        self.gt_dropout = gt_dropout
        self.gt_encoder_attention_heads = gt_encoder_attention_heads
        self.gt_encoder_ffn_dim = gt_encoder_ffn_dim
        self.gt_encoder_layerdrop = gt_encoder_layerdrop
        self.gt_encoder_layers = gt_encoder_layers
        self.gt_init_std = gt_init_std
        self.gt_is_encoder_decoder = gt_is_encoder_decoder
        self.min_length = min_length
        self.gt_normalize_before = gt_normalize_before
        self.gt_normalize_embedding = gt_normalize_embedding
        self.gt_scale_embedding = gt_scale_embedding
        self.num_beams = num_beams
        self.max_length = max_length
        self.rezero = rezero


'''
# 实现了图 Transformer 的多头注意力层
class GraphTransformerMultiHeadAttentionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.hid_dim = config.d_model  # 获取模型的隐藏层维度1024
        self.n_heads = config.gt_encoder_attention_heads  # 获取多头注意力的头数8

        assert self.hid_dim % self.n_heads == 0
        self.head_dim = self.hid_dim // self.n_heads  # 计算每个头的维度128

        self.fc_q = nn.Linear(self.hid_dim, self.hid_dim)  # 定义查询（Q）、键（K）、值（V）的线性变换层
        self.fc_k = nn.Linear(self.hid_dim, self.hid_dim)
        self.fc_v = nn.Linear(self.hid_dim, self.hid_dim)

        self.fc_o = nn.Linear(self.hid_dim, self.hid_dim)  # 定义输出的线性变换层，将多头注意力的结果进行整合
        self.dropout = nn.Dropout(config.gt_attention_dropout)  # 定义丢弃层，用于防止过拟合0.1
        self.scale = self.head_dim ** 0.5  # 计算缩放因子，用于缩放注意力分数

        self.feature_maps_num = len(config.feature_types)  # 获取特征类型的数量4+1
        self.feature_conv = nn.Sequential(*[nn.Conv2d(
            in_channels=self.feature_maps_num,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
        ) for _ in range(self.n_heads)])  # 定义特征卷积层列表，每个卷积层将多个特征图合并为一个，卷积核大小为 1x1
        self.combine_conv = nn.Conv2d(  # 定义合并卷积层，将两个特征图合并为一个
            in_channels=2,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.conv_activation_fn = name_2_activation_fn_mapping[config.conv_activation_fn]  # 根据配置中的激活函数名称，获取对应的激活函数

        if 'topic_adj' in config.feature_types:  # 新增主题邻接矩阵嵌入
            self.topic_embed = nn.Embedding(2, 1, padding_idx=0)
        if 'discourse_adj' in config.feature_types:  # 如果特征类型中包含 'discourse_adj'，定义对话关系嵌入层
            self.discourse_embed = nn.Embedding(len(DISCOURSE_RELATIONS) + 1, 1,
                                                padding_idx=0)  # 嵌入层的输入维度为对话关系的数量加 1，输出维度为 1
        if 'distance_adj' in config.feature_types:
            self.distance_embed = nn.Embedding(config.max_utt_num * 2 - 1, 1)
        if "cooccurrence_adj" in config.feature_types:
            self.cooccurrence_embed = nn.Embedding(5 + 1, 1, padding_idx=0)  # [0,1,2,3,4]
        if "Redundancy_adj" in config.feature_types:  # 新增：冗余嵌入
            self.redundancy_embed = nn.Embedding(2, 1, padding_idx=0)  # 二值嵌入（0表示冗余，1表示非冗余）
        # 新增用于注意力融合的权重计算
        #self.weight_dynamic = nn.Linear(self.hid_dim, self.n_heads)

    def forward(self, query, key, value, adj_mats, mask):
        batch_size = query.shape[0]
        Q = self.fc_q(query)  # bs,seq_len,hid_dim
        K = self.fc_k(key)
        V = self.fc_v(value)

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # [bs,n_heads,seq_len,hid_dim]
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        # energy = [batch size, n heads, query len, key len]

        adj_ls = []
        for k, v in adj_mats.items():
            if k == 'distance_adj':
                adj_ls.append(self.distance_embed(v).squeeze(-1))  # [bs,num_utt,num_utt]
            elif k == 'speaker_adj':
                adj_ls.append(v)  # [bs,num_utt,num_utt]
            elif k == 'topic_adj':  # 新增处理逻辑
                adj_ls.append(self.topic_embed(v.long()).squeeze(-1))
            elif k == 'discourse_adj':
                adj_ls.append(self.discourse_embed(v).squeeze(-1))
            elif k == "cooccurrence_adj":
                adj_ls.append(self.cooccurrence_embed(v).squeeze(-1))
            elif k == 'Redundancy_adj':  # 新增：处理冗余邻接矩阵
                adj_ls.append(self.redundancy_embed(v.long()).squeeze(-1))  # [bs, num_utt, num_utt]
        # [bs,feature_map_num,num_utt,num_utt]
        feature_map = torch.stack(adj_ls, dim=1)

        # feature_conv_output: [bs,n_heads,q_len,k_len]
        feature_conv_output = self.conv_activation_fn(
            torch.stack([conv(feature_map).squeeze(1) for conv in self.feature_conv], dim=1))

        energy_ls = []
        for idx in range(self.n_heads):
            energy_ls.append(self.combine_conv(
                torch.stack([energy[:, idx, :, :], feature_conv_output[:, idx, :, :]], dim=1)).squeeze(1))
        energy = self.conv_activation_fn(torch.stack(energy_ls, dim=1))  # bs,n_head,q_len,k_len

        if mask is not None:
            # _energy = _energy.masked_fill(mask==0,float("-inf"))
            energy = energy.masked_fill(mask == 0, float("-inf"))
            # feature_conv_output = feature_conv_output.masked_fill(mask==0,float("-inf"))

        attention = torch.softmax(energy, dim=-1)
        # feature_conv_output = torch.softmax(feature_conv_output,dim=-1)
        # feature_conv_output = torch.softmax(feature_conv_output,dim=-1)
        # energy = [batch size, n heads, query len, key len]

        # p_dyna = torch.stack([torch.sigmoid(self.p_dyna_linear[x](Q[:,x,:,:])) for x in range(self.n_heads)],dim=1) # bs,n_heads,seq_len,1
        # attention = p_dyna * energy + (1-p_dyna) * feature_conv_output
        # if self.config.ablation_type == "static":
        #    attention = torch.softmax(feature_conv_output,dim=-1)
        # elif self.config.ablation_type == 'dynamic':
        #    attention = torch.softmax(_energy,dim=-1)

        x = torch.matmul(self.dropout(attention), V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.hid_dim)
        x = self.fc_o(x)
        return x, attention
'''
'''
    def forward(self, query, key, value, adj_mats, mask):
        batch_size = query.shape[0]
        #hidden_states = query  # 使用 query 作为隐藏状态
        #num_utt = hidden_states.size(1)  # 获取话语数量

        Q = self.fc_q(query)  # 通过线性变换层得到查询、键、值的表示#bs,seq_len,hid_dim(1,5,1024)
        K = self.fc_k(key)
        V = self.fc_v(value)
        # 将查询、键、值的表示分割成多个头，并调整维度
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1,
                                                                        3)  # [bs,n_heads,seq_len,hid_dim](1,8,5,128)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale  # 计算注意力分数，通过矩阵乘法和缩放操作(1,8,5,5)

        adj_ls = []  # 用于存储不同类型邻接矩阵处理后的结果
        for k, v in adj_mats.items():
            if k == 'distance_adj':  # 对距离邻接矩阵进行嵌入处理，并去除最后一维
                adj_ls.append(self.distance_embed(v).squeeze(-1))  # [bs,num_utt,num_utt]
            elif k == 'speaker_adj':  # 直接添加说话人邻接矩阵
                adj_ls.append(v)  # [bs,num_utt,num_utt]
            elif k == 'discourse_adj':  # 对话语关系邻接矩阵进行嵌入处理，并去除最后一维
                adj_ls.append(self.discourse_embed(v).squeeze(-1))
            elif k == "cooccurrence_adj":  # 对共现邻接矩阵进行嵌入处理，并去除最后一维
                adj_ls.append(self.cooccurrence_embed(v).squeeze(-1))
            elif k == 'topic_adj':  # 新增处理逻辑
                adj_ls.append(self.topic_embed(v.long()).squeeze(-1))
        # [bs,feature_map_num,num_utt,num_utt]
        feature_map = torch.stack(adj_ls, dim=1)  # 将不同类型的邻接矩阵处理结果在维度 1 上堆叠，形成特征图(1,4,5,5)

        # feature_conv_output: [bs,n_heads,q_len,k_len] 对特征图应用卷积操作，并通过激活函数处理，得到每个头的特征卷积输出
        feature_conv_output = self.conv_activation_fn(torch.stack([conv(feature_map).squeeze(1) for conv in self.feature_conv], dim=1))



        # 计算上下文向量用于动态调整权重
        context_vector = hidden_states.mean(dim=1)  # [bs, hid_dim]
        weight_dynamic = torch.sigmoid(self.weight_dynamic(context_vector))  # [bs, n_heads]
        weight_dynamic = weight_dynamic.unsqueeze(2).unsqueeze(3)  # [bs, n_heads, 1, 1]
        weight_static = 1 - weight_dynamic
        # 扩展权重以匹配 energy 和 feature_conv_output 的维度
        weight_dynamic = weight_dynamic.expand(-1, -1, num_utt, num_utt)  # [bs, n_heads, num_utt, num_utt]
        weight_static = weight_static.expand(-1, -1, num_utt, num_utt)  # [bs, n_heads, num_utt, num_utt]
        # 使用权重融合动态和静态部分
        combined_energy = weight_dynamic * energy + weight_static * feature_conv_output
        energy_ls = []  # 用于存储每个头合并后的注意力能量
        for idx in range(self.n_heads):  # 将原始注意力能量和特征卷积输出在维度 1 上堆叠，通过合并卷积层处理，并去除维度 1
            energy_ls.append(self.combine_conv(torch.stack([combined_energy[:, idx, :, :], feature_conv_output[:, idx, :, :]], dim=1)).squeeze(1))
        energy = self.conv_activation_fn(torch.stack(energy_ls, dim=1))  # bs,n_head,q_len,k_len(1,8,5,5) 将每个头合并后的注意力能量在维度 1 上堆叠，并通过激活函数处理


        energy_ls = []
        for idx in range(self.n_heads):
            energy_ls.append(self.combine_conv(
               torch.stack([energy[:, idx, :, :], feature_conv_output[:, idx, :, :]], dim=1)).squeeze(1))
        energy = self.conv_activation_fn(torch.stack(energy_ls, dim=1))  # bs,n_head,q_len,k_len


        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-inf"))  # 将掩码为 0 的位置的注意力能量置为负无穷，在 softmax 计算时会趋近于 0

        attention = torch.softmax(energy, dim=-1)  # 对注意力能量应用 softmax 函数，得到注意力分数

        x = torch.matmul(self.dropout(attention), V)  # 通过注意力分数和值进行矩阵乘法，并应用丢弃操作
        x = x.permute(0, 2, 1, 3).contiguous()  # 调整维度，将头的维度和序列长度维度交换，并将张量连续化
        x = x.view(batch_size, -1, self.hid_dim)  # 将多个头的结果合并
        x = self.fc_o(x)  # 通过输出的线性变换层得到最终输出
        return x, attention  # x 是融合了输入序列中不同位置之间的依赖关系以及图结构信息后的特征表示
'''


class GraphTransformerMultiHeadAttentionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.hid_dim = config.d_model  # 隐藏层维度1024
        self.n_heads = config.gt_encoder_attention_heads  # 多头注意力头数8

        assert self.hid_dim % self.n_heads == 0
        self.head_dim = self.hid_dim // self.n_heads  # 每个头的维度128

        self.fc_q = nn.Linear(self.hid_dim, self.hid_dim)  # 查询、键、值线性变换层
        self.fc_k = nn.Linear(self.hid_dim, self.hid_dim)
        self.fc_v = nn.Linear(self.hid_dim, self.hid_dim)

        self.fc_o = nn.Linear(self.hid_dim, self.hid_dim)  # 输出线性变换层
        self.dropout = nn.Dropout(config.gt_attention_dropout)  # 丢弃层0.1
        self.scale = self.head_dim ** 0.5  # 缩放因子

        self.feature_maps_num = len(config.feature_types)  # 特征类型数量（5，包括冗余图）
        self.feature_conv = nn.Sequential(*[nn.Conv2d(
            in_channels=self.feature_maps_num,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
        ) for _ in range(self.n_heads)])  # 特征卷积层
        self.combine_conv = nn.Conv2d(  # 合并卷积层
            in_channels=2,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.conv_activation_fn = name_2_activation_fn_mapping[config.conv_activation_fn]  # 激活函数

        # 嵌入层定义
        if 'topic_adj' in config.feature_types:
            self.topic_embed = nn.Embedding(2, 1, padding_idx=0)
        if 'discourse_adj' in config.feature_types:
            self.discourse_embed = nn.Embedding(len(DISCOURSE_RELATIONS) + 1, 1, padding_idx=0)
        if 'distance_adj' in config.feature_types:
            self.distance_embed = nn.Embedding(config.max_utt_num * 2 - 1, 1)
        if "cooccurrence_adj" in config.feature_types:
            self.cooccurrence_embed = nn.Embedding(5 + 1, 1, padding_idx=0)
        if "Redundancy_adj" in config.feature_types:  # 优化：增强冗余嵌入
            self.redundancy_embed = nn.Embedding(2, 1, padding_idx=0)  # 输出维度提升至head_dim（32）
            #self.redundancy_weight = nn.Parameter(torch.tensor(0.5))  # 可学习权重

    def forward(self, query, key, value, adj_mats, mask):
        batch_size = query.shape[0]
        num_utt = query.size(1)  # 获取话语数量

        Q = self.fc_q(query)  # [bs, seq_len, hid_dim]
        K = self.fc_k(key)
        V = self.fc_v(value)

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # [bs, n_heads, seq_len, head_dim]
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale  # [bs, n_heads, seq_len, seq_len]

        adj_ls = []
        for k, v in adj_mats.items():
            if k == 'distance_adj':
                adj_ls.append(self.distance_embed(v).squeeze(-1))  # [bs, num_utt, num_utt]
            elif k == 'speaker_adj':
                adj_ls.append(v)
            elif k == 'topic_adj':
                adj_ls.append(self.topic_embed(v.long()).squeeze(-1))
            elif k == 'discourse_adj':
                adj_ls.append(self.discourse_embed(v).squeeze(-1))
            elif k == "cooccurrence_adj":
                adj_ls.append(self.cooccurrence_embed(v).squeeze(-1))
            elif k == 'Redundancy_adj':  # 优化：增强冗余图处理
                adj_ls.append(self.redundancy_embed(v.long()).squeeze(-1))

        feature_map = torch.stack(adj_ls, dim=1)  # [bs, feature_maps_num, num_utt, num_utt]
        feature_conv_output = self.conv_activation_fn(
            torch.stack([conv(feature_map).squeeze(1) for conv in self.feature_conv],
                        dim=1))  # [bs, n_heads, num_utt, num_utt]

        # 优化：直接屏蔽冗余语句
        if 'Redundancy_adj' in adj_mats:
            redundancy_mask = (adj_mats['Redundancy_adj'] == 0).unsqueeze(1)  # [bs, 1, num_utt, num_utt]
            feature_conv_output = feature_conv_output.masked_fill(redundancy_mask, -10)  # 屏蔽冗余语句
            energy = energy.masked_fill(redundancy_mask, -15)
        energy_ls = []
        for idx in range(self.n_heads):
            energy_ls.append(self.combine_conv(
                torch.stack([energy[:, idx, :, :], feature_conv_output[:, idx, :, :]], dim=1)).squeeze(1))
        energy = self.conv_activation_fn(torch.stack(energy_ls, dim=1))  # [bs, n_heads, num_utt, num_ut
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-inf"))  # 应用常规掩码

        attention = torch.softmax(energy, dim=-1)  # [bs, n_heads, num_utt, num_utt]
        x = torch.matmul(self.dropout(attention), V)  # [bs, n_heads, num_utt, head_dim]
        x = x.permute(0, 2, 1, 3).contiguous()  # [bs, num_utt, n_heads, head_dim]
        x = x.view(batch_size, -1, self.hid_dim)  # [bs, num_utt, hid_dim]
        x = self.fc_o(x)  # [bs, num_utt, hid_dim]
        return x, attention


# 用于构建图 Transformer 的单个编码层
class GraphTransformerLayer(BartEncoderLayer):
    # GraphTransformerLayer 类继承自 BartEncoderLayer，用于构建图 Transformer 模型中的单个层，该层结合了针对图结构数据的多头注意力机制和前馈神经网络，以处理图相关的输入。
    def __init__(self, config):
        super().__init__(config)
        self.self_attn = GraphTransformerMultiHeadAttentionLayer(
            config)  # GraphTransformerMultiHeadAttentionLayer 实现了针对图结构数据的多头注意力机制
        self.fc1 = nn.Linear(self.embed_dim, config.gt_encoder_ffn_dim)  # 定义前馈神经网络的第一个线性层，（1024，4096）
        self.fc2 = nn.Linear(config.gt_encoder_ffn_dim, self.embed_dim)  # 定义前馈神经网络的第二个线性层，（4096，1024）

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor,  # tensor([[[[True, True, True, True, True]]]], device='cuda:0')
            layer_head_mask: torch.Tensor,
            output_attentions: bool = False,
            **kwargs,
    ):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (:obj:`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        adj_mats = kwargs.get('adj_mats', None)  # 从 kwargs 中获取邻接矩阵，如果未提供则为 None
        residual = hidden_states  # 保存输入的隐藏状态作为残差连接的输入
        hidden_states, attn_weights = self.self_attn(  # 调用自定义的多头注意力层进行计算，输入为查询、键、值，以及邻接矩阵和注意力掩码
            hidden_states, hidden_states, hidden_states,  # 返回更新后的隐藏状态和注意力权重
            adj_mats,
            mask=attention_mask,  # attention_mask = tensor([[[[True, True, True, True, True]]]], device='cuda:0')
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout,
                                              training=self.training)  # 对更新后的隐藏状态应用丢弃（dropout）操作，以一定的概率 p（self.dropout）随机将部分元素置为 0，防止模型过拟合
        hidden_states = residual + hidden_states  # 将更新后的隐藏状态与残差连接相加，缓解梯度消失问题
        hidden_states = self.self_attn_layer_norm(hidden_states)  # 对相加后的隐藏状态进行层归一化处理，使输入的特征分布更加稳定

        residual = hidden_states  # 保存经过多头注意力和层归一化后的隐藏状态作为残差连接的输入
        hidden_states = self.activation_fn(self.fc1(hidden_states))  # 将隐藏状态输入到前馈神经网络的第一个线性层，并通过激活函数进行非线性变换
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout,
                                              training=self.training)  # 对经过激活函数处理后的隐藏状态应用丢弃操作，丢弃率为0.1
        hidden_states = self.fc2(hidden_states)  # 将经过丢弃操作后的隐藏状态输入到前馈神经网络的第二个线性层
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout,
                                              training=self.training)  # 再次对隐藏状态应用丢弃操作，丢弃率为0.1
        hidden_states = residual + hidden_states  # 将经过前馈神经网络处理后的隐藏状态与残差连接相加
        hidden_states = self.final_layer_norm(hidden_states)  # 对相加后的隐藏状态进行最终的层归一化处理

        if hidden_states.dtype == torch.float16 and (  # 检查隐藏状态的数据类型是否为 torch.float16，并且是否存在无穷大或 NaN 值
                torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000  # 如果存在无穷大或 NaN 值，将隐藏状态的值限制在一个范围内
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)  # 初始化输出元组，包含更新后的隐藏状态

        if output_attentions:
            outputs += (attn_weights,)  # 如果需要输出注意力分数，将注意力权重添加到输出元组中

        return outputs


# 它实现了一个图 Transformer 编码器。该编码器会根据配置创建指定数量的 GraphTransformerLayer 层，并且可以根据配置选择不同的位置嵌入方式
class GraphTransformer(BartEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList([GraphTransformerLayer(config) for _ in range(
            int(config.gt_encoder_layers))])  # 每个图Transformer层是一个 GraphTransformerLayer 实例.4层
        del self.embed_positions  # 因为图Transformer可能需要不同的位置嵌入方式，所以将父类的默认位置嵌入层删除
        if config.gt_pos_embed == 'learned':  # 根据配置中的 gt_pos_embed 参数选择不同的位置嵌入方式
            self.embed_positions = nn.Embedding(1026, config.d_model)
        elif config.gt_pos_embed == 'sinusoidal':
            self.embed_positions = SinusoidalPositionalEmbedding(1024, config.d_model)

    # 根据 return_dict 的值，返回元组或 BaseModelOutput 对象，包含最后一层隐藏状态、各层隐藏状态和注意力分数。
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
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions  # 如果未提供 output_attentions 参数，则使用配置中的 output_attentions 设置
        output_hidden_states = (  # 如果未提供 output_hidden_states 参数，则使用配置中的 output_hidden_states 设置
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 检查输入，input_ids 和 inputs_embeds 不能同时提供
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])  # 将输入的形状调整为二维
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]  # 从输入嵌入中获取输入的形状（不包括嵌入维度）（1，5）
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale  # 如果没有提供输入嵌入，则通过词嵌入层将输入 ID 转换为嵌入表示，并进行缩放
        bs, num_utt = input_shape
        if hasattr(self, 'embed_positions'):
            if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):  # 如果存在位置嵌入层
                embed_pos = self.embed_positions(input_shape)  # 如果是正弦位置嵌入层，调用其前向传播方法获取位置嵌入
            else:
                embed_pos = self.embed_positions(
                    torch.arange(num_utt).view(1, -1).repeat(bs, 1).to(inputs_embeds.device))  # 否则，通过索引获取位置嵌入
            hidden_states = inputs_embeds + embed_pos  # 将输入嵌入和位置嵌入相加得到隐藏状态
        else:
            hidden_states = inputs_embeds  # 如果没有位置嵌入层，直接使用输入嵌入作为隐藏状态

        hidden_states = self.layernorm_embedding(hidden_states)  # 对隐藏状态进行层归一化处理，使输入的特征分布更加稳定
        hidden_states = F.dropout(hidden_states, p=self.dropout,
                                  training=self.training)  # 对隐藏状态应用丢弃（dropout）操作，以一定的概率 p（self.dropout）随机将部分元素置为 0，防止模型过拟合

        # 扩展注意力掩码
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = (attention_mask == 1).unsqueeze(1).unsqueeze(2)  # 将注意力掩码转换为布尔类型，并在第 1 和第 2 维上进行扩展，

        encoder_states = () if output_hidden_states else None  # 如果需要输出各层隐藏状态，初始化一个空元组用于存储；否则设为 None
        all_attentions = () if output_attentions else None  # 如果需要输出注意力分数，初始化一个空元组用于存储；否则设为 None

        # # 检查头掩码的层数是否与模型的层数一致
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, encoder_layer in enumerate(self.layers):  # 遍历每一层图 Transformer 层
            if output_hidden_states:  # 如果需要输出各层隐藏状态，将当前隐藏状态添加到 encoder_states 中
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)  # 生成一个 0 到 1 之间的随机浮点数，用于 LayerDrop 机制
            if self.training and (dropout_probability < self.layerdrop):  # 如果处于训练模式且随机数小于层丢弃率，跳过当前层，将层输出设为 (None, None)
                layer_outputs = (None, None)
            else:
                if getattr(self.config, "gradient_checkpointing", False) and self.training:  # 如果启用了梯度检查点机制且处于训练模式
                    # 定义一个自定义的前向传播函数
                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    # 使用梯度检查点进行计算，以节省内存
                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                else:
                    layer_outputs = encoder_layer(  # 正常调用图 Transformer 层的前向传播方法
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                        **kwargs,
                    )

                hidden_states = layer_outputs[0]  # 更新隐藏状态

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)  # 如果需要输出注意力分数，将当前层的注意力分数添加到 all_attentions 中

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)  # 如果需要输出各层隐藏状态，将最后一层的隐藏状态添加到 encoder_states 中

        if not return_dict:  # 如果不使用字典形式返回结果，将隐藏状态、各层隐藏状态和注意力分数组合成元组返回
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(  # 如果使用字典形式返回结果，返回一个 BaseModelOutput 对象
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class RGCN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.graph_layers = nn.ModuleList(
            [dglnn.RelGraphConv(
                in_feat=config.d_model,
                out_feat=config.d_model,
                num_rels=17,
                regularizer='basis',
                self_loop=False,
            )
                for _ in range(3)]
        )

    def forward(self, inputs_embeds, attention_mask, **kwargs):
        input = inputs_embeds
        # input: bs,num_utt,num_utt
        # attn_mask: bs,num_utt
        adj_mats = kwargs.get('adj_mats', None)
        discourse_adj = adj_mats['discourse_adj']  # bs,num_utt,num_utt
        graph_output = []
        for batch_idx in range(input.size()[0]):
            src, trg, etype = [], [], []
            mat = discourse_adj[batch_idx].tolist()
            mat_len = len(mat)
            for i in range(mat_len):
                for j in range(mat_len):
                    if mat[i][j] != 0:
                        src.append(i)
                        trg.append(j)
                        etype.append(mat[i][j])
            graph = dgl.graph((torch.tensor(src), torch.tensor(trg)), num_nodes=mat_len).to(input.device)
            etype = torch.tensor(etype, device=input.device, dtype=torch.int64)
            f_in = input[batch_idx]
            for layer in self.graph_layers:
                f_in = layer(graph, f_in, etype)
            graph_output.append(f_in)
        graph_output = torch.stack(graph_output, dim=0)
        assert graph_output.size() == input.size()
        return BaseModelOutput(last_hidden_state=graph_output)

