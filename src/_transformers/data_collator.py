import random
from re import X
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

import torch
from torch.nn.utils.rnn import pad_sequence

from transformers.file_utils import PaddingStrategy
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

InputDataClass = NewType("InputDataClass", Any)

@dataclass
class MyDataCollatorForSeq2Seq:
    
    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    dis_offset: int = 463
    adj_mat_ls: list= None

    def __call__(self,features):
        
        # features List[dict]
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # 在调用 tokenizer.pad 方法之前，需要先对标签进行填充，因为该方法不会自动填充标签，
        # 并且需要标签具有相同的长度才能返回张量
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                feature["labels"] = (
                    feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                )
        # if self.max_utt_threshold > 0:
        #     for feature in feature:
        #         for n in ['gt_input_ids','gt_attention_mask','']
        # 如果特征中包含 'gt_input_ids' 字段，说明需要处理图相关的输入
        if 'gt_input_ids' in features[0].keys():
            # 提取普通特征（不包含图相关特征
            normal_features_ls = [
                {'attention_mask':x['attention_mask'],'input_ids':x['input_ids'],'labels':x['labels']} for x in features
                                 ]
            normal_padded_featurs = self.tokenizer.pad(
                normal_features_ls,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )       # 使用分词器对普通特征进行填充
            # 初始化填充后的特征字典，用于存储图相关特征,调用 pad_list_of_tensor 函数对图输入 ID 和注意力掩码进行填充
            padded_features = {'gt_input_ids':[],"gt_attention_mask":[],"adj_mats":[]}
            padded_features['num_utt_ls'],padded_features['gt_input_ids'],padded_features['gt_attention_mask'] = pad_list_of_tensor(features,self.tokenizer)
            max_num_utt = max([len(feature['gt_input_ids']) for feature in features])       # 计算最大话语数量

            adj_mats_ls = self.adj_mat_ls   # 获取邻接矩阵类型列表
            if any([m in features[0].keys() for m in adj_mats_ls]): padded_features['adj_mats'] = {}    # 如果特征中包含任何邻接矩阵类型，则初始化 adj_mats 字典
            for adj_type in adj_mats_ls:
                if adj_type in features[0].keys():
                    padded_features['adj_mats'][adj_type] = []

            for feature in features:
                 for adj_type in adj_mats_ls:
                    if adj_type in feature.keys():
                        mat = feature[adj_type]
                        ori_mat_size = len(mat)
                        mat = torch.tensor(mat)     # 将邻接矩阵转换为张量
                        if not ori_mat_size <= max_num_utt:
                            print(adj_type)
                        padded_mat = torch.zeros((max_num_utt,max_num_utt),dtype=mat.dtype)     # 初始化填充后的邻接矩阵
                        padded_mat[:ori_mat_size,:ori_mat_size] = mat       # 将原始邻接矩阵复制到填充后的矩阵中
                        if adj_type == 'distance_adj':          # 如果是距离邻接矩阵，添加距离偏移量
                            padded_mat += self.dis_offset
                    padded_features['adj_mats'][adj_type].append(padded_mat) # 将填充后的邻接矩阵添加到对应的列表中#(5,5)
            # 如果 adj_mats 字典不为空，将每个邻接矩阵列表转换为张量
            if 'adj_mats' in padded_features.keys():
                for k,v in padded_features['adj_mats'].items():
                    padded_features['adj_mats'][k] = torch.stack(v)             #(1,5,5)
            # 将普通特征的填充结果添加到最终的填充特征字典中
            padded_features['input_ids'] = normal_padded_featurs['input_ids']
            padded_features['attention_mask'] = normal_padded_featurs['attention_mask']
            padded_features['labels'] = normal_padded_featurs['labels']
        else: 
            padded_features = self.tokenizer.pad(
                features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"): # 如果提供了模型，并且模型具有 prepare_decoder_input_ids_from_labels 方法
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=padded_features["labels"])      # 则根据标签生成解码器输入 ID
            padded_features ["decoder_input_ids"] = decoder_input_ids   # 在解码器生成目标序列的过程中，它需要逐步预测每个词元（token），而 decoder_input_ids 就是解码器在每一步接收的输入
         
        return padded_features 


def pad_list_of_tensor(features,tokenizer):
    """
    输入的特征列表，每个元素是一个字典，包含图输入 ID 和注意力掩码,tuple: 包含每个样本的话语数量列表、填充后的图输入 ID 张量和填充后的注意力掩码张量
    """
    max_seq_len_in_a_batch = 0
    len_ls = []
    for sample in features:
        len_ls.append(len(sample['gt_input_ids']))
        for utt in sample['gt_input_ids']:
            l = len(utt)
            if l > max_seq_len_in_a_batch:
                max_seq_len_in_a_batch = l
    
    for sample in features:
        for k,v in sample.items():
            if k=='gt_input_ids':
                for utt in v:
                    diff = max_seq_len_in_a_batch - len(utt)
                    utt += [tokenizer.pad_token_id] * diff # utt = utt + is wrong
            elif k=='gt_attention_mask':
                for mask in v:
                    diff = max_seq_len_in_a_batch - len(mask)
                    mask += [0] * diff
    return len_ls,torch.cat([torch.tensor(x['gt_input_ids']) for x in features],dim=0),torch.cat([torch.tensor(x['gt_attention_mask']) for x in features],dim=0)


'''def _build_word_graph(self, sentences):
    """生成词级依存句法邻接矩阵"""
    import spacy
    nlp = spacy.load("en_core_web_sm")

    word_adj = []
    for sent in sentences:
        doc = nlp(sent)
        size = len(doc)
        adj = np.zeros((size, size))

        # 添加依存关系边
        for token in doc:
            if token.head.i != token.i:
                adj[token.i][token.head.i] = 1
                adj[token.head.i][token.i] = 1  # 双向连接

        # 添加共现关系边（窗口=3）
        for i in range(size):
            for j in range(max(0, i - 3), min(size, i + 3)):
                if i != j: adj[i][j] += 0.5

        word_adj.append(adj)
    return word_adj
'''

'''def _build_phrase_graph(self, sentences):
    """生成短语级语义角色邻接矩阵"""
    from allennlp.predictors import Predictor
    predictor = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/bert-base-srl-2020.11.19.tar.gz")

    phrase_adj = []
    for sent in sentences:
        result = predictor.predict(sentence=sent)
        verbs = result["verbs"]
        size = len(verbs)
        adj = np.zeros((size, size))

        # 基于相同语义角色建立连接
        for v in verbs:
            args = [i for i, tag in enumerate(v["tags"]) if "ARG" in tag]
            for i, j in combinations(args, 2):
                adj[i][j] = 1
                adj[j][i] = 1

        phrase_adj.append(adj)
    return phrase_adj
'''
'''
def _build_cross_links(self, sentences):
    """构建词→短语→句子的跨层连接"""
    cross_links = []
    for sent in sentences:
        # 简化的映射逻辑（需根据实际分析工具优化）
        words = sent.split()
        phrases = ["NP"] * len(words)  # 示例伪代码
        links = {
            "word2phrase": [i % len(phrases) for i in range(len(words))],
            "phrase2sent": [0] * len(phrases)
        }
        cross_links.append(links)
    return cross_links
'''