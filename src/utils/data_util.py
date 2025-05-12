import enum
from datasets import dataset_dict, load_dataset,DatasetDict
from networkx.readwrite.json_graph import adjacency
from transformers import  DataCollatorForSeq2Seq
import nltk
import logging
from tqdm import tqdm
import torch
import stanza
#logger = logging.getLogger(__name__)
import os 

# own
from src.utils.CONSTANT import DISCOURSE_RELATIONS


def get_dataset(data_args,model_args):
    
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir)
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        # 使用 load_dataset 函数从本地文件加载数据集
        datasets = load_dataset(extension, data_files=data_files)
        
    return datasets
    
def data_preprocessing(datasets,tokenizer,training_args,data_args,model_args):
    
    # if data_args.save_dataset_path is None or data_args.reprocess_data:
    if training_args.do_train:
        column_names = datasets["train"].column_names
    elif training_args.do_eval:
        column_names = datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = datasets["test"].column_names

    # 从数据参数中获取文本列名、摘要列名、说话者列名和唯一说话者列名
    text_column = data_args.text_column
    summary_column = data_args.summary_column
    speaker_column = data_args.speaker_column
    unique_speaker_column = data_args.unique_speaker_column  # maybe we should dynamically add new column?

    padding = "max_length" if data_args.pad_to_max_length else False        # 根据配置决定是否对输入进行填充到最大长度
       
    def preprocess_function(examples):
        # examples:{str:[List[List]]}  # 从 examples 中提取源文本、目标文本、说话者信息、唯一说话者信息、样本 ID、话语关系和关键词信息
        source = examples[text_column] # List[List[str]]
        targets = examples[summary_column] #  List[str]
        speakers = examples[speaker_column] # List[List[str]]
        unique_speakers = examples[unique_speaker_column] #List[List[str]]
        ids = examples['id']
        discourse_relations = examples['discourse_relations'] if 'discourse_relations' in examples.keys() else None
        topic_segments = examples['TS']  # 获取主题分割信息
        keywords = examples['keywords'] if 'keywords' in examples.keys() else None
        Redundancy = examples['RD'] if 'RD' in examples.keys() else None  # 新增：提取冗余性标注
        bs = len(source)    # 获取当前批次的样本数量
        if model_args.model_type == 'graphtransformer':
            # 初始化模型输入字典，用于存储图 Transformer 模型的输入数据
            model_inputs = {'gt_input_ids': [],
                            'gt_attention_mask': [],
                            'id':[]}
            # 根据特征类型，在模型输入字典中添加相应的图特征键
            if "distance_adj" in model_args.feature_types:
                model_inputs['distance_adj'] = []
            if "speaker_adj" in model_args.feature_types:
                model_inputs['speaker_adj'] = []
            if "discourse_adj" in model_args.feature_types:
                model_inputs['discourse_adj'] = []
            if "cooccurrence_adj" in model_args.feature_types:
                model_inputs['cooccurrence_adj'] = []
            if "topic_adj" in model_args.feature_types:  # 添加主题分割邻接矩阵
                model_inputs['topic_adj'] = []
            if "Redundancy_adj" in model_args.feature_types:
                model_inputs['Redundancy_adj'] = []
            inputs = []
            for batch_idx in range(bs):
                utts = source[batch_idx] #List[str]
                # 判断 source_prefix 不是 None 且有内容时才添加
                if data_args.source_prefix:
                    utts = [f"{data_args.source_prefix}: {utt}" for utt in utts]
                if discourse_relations:
                    discourse_relations_per_instance = discourse_relations[batch_idx] #List[List[int,str,int]]
                if keywords:
                    keywords_per_instance = keywords[batch_idx]

                # 如果源文本的话语数量超过最大允许数量30，进行截断
                if len(utts) > data_args.max_utt_num:
                    utts = utts[:data_args.max_utt_num]
                    unique_speakers[batch_idx] = unique_speakers[batch_idx][:data_args.max_utt_num]
                    speakers[batch_idx] = speakers[batch_idx][:data_args.max_utt_num]
                # 对当前样本的话语进行分词处理
                tokenized_utts = tokenizer(utts, max_length=data_args.max_seq_len_per_utt, padding=padding, truncation=True)
                model_inputs['gt_input_ids'].append(tokenized_utts['input_ids'])    # 将分词后的输入 ID 和注意力掩码添加到模型输入字典中
                model_inputs['gt_attention_mask'].append(tokenized_utts['attention_mask'])
                model_inputs['id'].append(ids[batch_idx])
                if "distance_adj" in model_args.feature_types:
                    model_inputs['distance_adj'].append(get_distance_adj(len(utts)))
                if "speaker_adj" in model_args.feature_types:
                    model_inputs['speaker_adj'].append(get_speaker_adj(unique_speakers[batch_idx]))
                if "discourse_adj" in model_args.feature_types:
                    model_inputs['discourse_adj'].append(get_discourse_adj(discourse_relations_per_instance,len(utts),data_args.max_utt_num))
                if "cooccurrence_adj" in model_args.feature_types:
                    model_inputs['cooccurrence_adj'].append(get_cooccurrence_adj(keywords_per_instance,len(utts)))
                if "Redundancy_adj" in model_args.feature_types:  # 新增：生成冗余性邻接矩阵
                    model_inputs['Redundancy_adj'].append(get_redundancy_adj(Redundancy, len(utts), data_args.max_utt_num))
                if "topic_adj" in model_args.feature_types:
                    if len(utts) != topic_segments[batch_idx][-1] + 1:
                        print(ids[batch_idx])
                        print(len(utts))
                        print(topic_segments[batch_idx])
                    topic_adj_matrix = get_topic_adj(topic_segments[batch_idx], len(utts))
                    model_inputs['topic_adj'].append(topic_adj_matrix)
                # 如果分词器中没有 '<sep>' 特殊标记，将其添加到分词器的特殊标记列表中
                if '<sep>' not in tokenizer.additional_special_tokens:
                    special_tokens_dict = {"additional_special_tokens":["<sep>"]}
                    tokenizer.add_special_tokens(special_tokens_dict)
                # 将当前样本的说话者和话语信息拼接成一个字符串
                inputs_str = ""
                for utt_idx in range(min(data_args.max_utt_num,len(source[batch_idx]))):
                    inputs_str += speakers[batch_idx][utt_idx]
                    inputs_str += ': '
                    inputs_str += source[batch_idx][utt_idx]
                    inputs_str += ' <sep> '
                inputs.append(inputs_str)
            baseline_input = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)    # 对拼接后的输入字符串进行分词处理
            model_inputs['input_ids'] = baseline_input['input_ids']     # 将分词后的输入 ID 和注意力掩码添加到模型输入字典中
            model_inputs['attention_mask'] = baseline_input['attention_mask']

        elif model_args.model_type == 'baseline':
            if '<sep>' not in tokenizer.additional_special_tokens:
                special_tokens_dict = {"additional_special_tokens":["<sep>"]}
                tokenizer.add_special_tokens(special_tokens_dict)
            
            inputs = []
            for batch_idx in range(bs):
                inputs_str = ""
                for utt_idx in range(len(source[batch_idx])):
                    
                    inputs_str += speakers[batch_idx][utt_idx]
                    inputs_str += ": "
                    inputs_str += source[batch_idx][utt_idx]
                    inputs_str += ' <sep> '
                inputs_str = inputs_str[:-7] # delete last <sep>
                inputs.append(inputs_str)
            model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)
        
        # 使用分词器对目标文本进行处理
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=data_args.max_target_length, padding=padding, truncation=True,add_special_tokens=False)
            for k,v in labels.items():
                if k == 'input_ids':
                    labels[k] = [x+[tokenizer.eos_token_id] for x in v]
                elif k == 'attention_mask':
                    labels[k] = [x+[1] for x in v]
        # 如果进行填充且忽略填充标记对损失的影响，将标签中的填充标记替换为 -100
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    output_datasets = [None,None,None]      # 初始化一个列表，用于存储预处理后的训练集、验证集和测试集
    ## dataset mappping
    if training_args.do_train:
        train_dataset = datasets["train"]
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        if data_args.max_train_samples is not None:     # 如果指定了最大训练样本数，对训练集进行采样
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        train_dataset = train_dataset.map(      # 对训练集应用预处理函数
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        output_datasets[0] = train_dataset

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        output_datasets[1] = eval_dataset

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
        predict_dataset = predict_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        output_datasets[2]=predict_dataset
    return output_datasets

def get_distance_adj(num_utt):
    
    ret = []
    for idx in range(num_utt):
        start = 0-idx
        row = []
        while len(row) < num_utt:
            row.append(start)
            start += 1
        ret.append(row)
    return ret

def get_speaker_adj(sp_ls,window_size=2,temperature=5):
    uni_sp_ls = list(set(sp_ls))
    uni_sp_num = len(uni_sp_ls)
    mat = torch.zeros((uni_sp_num,uni_sp_num))
    for idx,sp in enumerate(sp_ls):
        row = uni_sp_ls.index(sp)
        for w in range(1,window_size+1):
            if idx + w < len(sp_ls):
                col = uni_sp_ls.index(sp_ls[idx+w])
                mat[row,col] += 1   

    row_softmax = torch.nn.functional.softmax(mat/temperature,dim=0)
    col_softmax = torch.nn.functional.softmax(mat/temperature,dim=1)
    speaker_softmax = (row_softmax * col_softmax).tolist()

    ret = [[0]*len(sp_ls) for _ in range(len(sp_ls))]
    for row in range(len(sp_ls)):
        for col in range(len(sp_ls)):
            _from = uni_sp_ls.index(sp_ls[row])
            _to = uni_sp_ls.index(sp_ls[col])
            ret[row][col] = speaker_softmax[_from][_to]
    return ret

def get_discourse_adj(discourse_relations,utt_num,max_utt_num):

    # this funtion return one instance per run

    # filter out utt out of max_utt_num
    ret = [[0]*utt_num for _ in range(utt_num)]
    if not discourse_relations:
        return ret
    discourse_relations = [[int(x.split(" ")[0]),x.split(" ")[1],int(x.split(" ")[2])] for x in discourse_relations] 
    discourse_relations = [x for x in discourse_relations if x[0] < max_utt_num and x[2] < max_utt_num]
    #ret = [[0]*utt_num for _ in range(utt_num)] # 0 is pad embedding
    for rel in discourse_relations:
        ret[rel[0]][rel[2]] = DISCOURSE_RELATIONS.index(rel[1])+1 # +1 for avoid padding index in graphtrans embedding
    return ret

def get_cooccurrence_adj(keywords,num_utts,threshold=5):

    key_word_ls = [set(x.split("@")) for x in keywords]
    ret = [[0]*num_utts for _ in range(num_utts)]
    for idx in range(num_utts):
        for jdx in range(idx+1,num_utts): # 去除i和i的关键词重现特征
            if key_word_ls[idx] != {""} and key_word_ls[jdx] != {""}:
                
                ret[idx][jdx] = min(len(key_word_ls[idx] & key_word_ls[jdx]),threshold)
                ret[jdx][idx] = ret[idx][jdx]

    return ret


def get_topic_adj(topic_segments, num_utt):
    # 创建一个全为0的矩阵
    ret = [[0] * num_utt for _ in range(num_utt)]

    # 确保 topic_segments 中的每个值都不会超出 num_utt - 1
    topic_segments = [min(seg, num_utt - 1) for seg in topic_segments]

    # 遍历每个主题区间，填充矩阵
    for topic_idx, end_sentence in enumerate(topic_segments):
        # 如果是第一个主题，start_sentence 从 0 开始，否则取前一个主题的结束位置
        start_sentence = 0 if topic_idx == 0 else topic_segments[topic_idx - 1] + 1

        # 防止 start_sentence 超过 end_sentence
        start_sentence = min(start_sentence, end_sentence)

        # 填充矩阵，表示句子 i 和句子 j 属于同一主题
        for i in range(start_sentence, end_sentence + 1):
            for j in range(start_sentence, end_sentence + 1):
                ret[i][j] = 1  # 句子 i 和句子 j 属于同一主题

    return ret


def get_redundancy_adj(Redundancy, utt_num, max_utt_num):
    """
    生成冗余性邻接矩阵，处理三种情况：[]（无冗余）、[16]（单冗余）、[16, 17]（冗余对）。
    冗余语句与其他所有语句的连接设为 0，使其在生成摘要时无用。

    Args:
        Redundancy: List，当前样本的冗余对，可能为 []、[16] 或 [16, 17]
        utt_num: int，当前对话的语句数量
        max_utt_num: int，最大允许的语句数量

    Returns:
        List[List[int]]，冗余性邻接矩阵，0表示与冗余语句的连接，1表示非冗余连接
    """
    # 初始化一个 utt_num x utt_num 的全1矩阵
    ret = [[1] * utt_num for _ in range(utt_num)]

    # 如果 Redundancy 为空或不是列表，返回全1矩阵
    if not Redundancy or not isinstance(Redundancy, list):
        return ret

    # 收集冗余语句的索引
    redundant_indices = set()

    # 根据 Redundancy 的长度处理
    if len(Redundancy) == 1:  # Redundancy = [16]
        try:
            i = int(Redundancy[0])
            if i < max_utt_num and i < utt_num:
                redundant_indices.add(i)
        except (ValueError, IndexError):
            pass  # 忽略无效值

    elif len(Redundancy) == 2:  # Redundancy = [16, 17]
        try:
            i, j = int(Redundancy[0]), int(Redundancy[1])
            if i < max_utt_num and j < max_utt_num and i < utt_num and j < utt_num:
                redundant_indices.add(i)
                redundant_indices.add(j)
        except (ValueError, IndexError):
            pass  # 忽略无效值

    # 将冗余语句的整行整列设为 0
    for idx in redundant_indices:
        for j in range(utt_num):
            ret[idx][j] = 0  # 行设为 0，包括对角线
            ret[j][idx] = 0  # 列设为 0，包括对角线

    return ret