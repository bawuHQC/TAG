import json
import math
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# my own module
from utils.training_util import ShowModelParamsCallBack,FineTuneCallBack #have to import comet_ml before torch


from torch.utils import data

import sys
from dataclasses import dataclass, field,asdict
from typing import Optional

import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
#from datasets import load_dataset, load_metric

import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)

from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version

# my own module
from utils.args_util import ModelArguments,DataTrainingArguments,check_args
from _transformers.data_collator import MyDataCollatorForSeq2Seq
from _transformers.seq2seq_trainer import Seq2SeqTrainer
from _transformers.training_args_seq2seq import Seq2SeqTrainingArguments
from utils import model_util,data_util,training_util
from utils.CONSTANT import *
import logging
from utils.metrics_util import get_bert_score,get_rouge_score,get_meteor_score

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.7.0.dev0")

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)
def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    config_file = './config/graphbart_config.json'
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):config_file = sys.argv[1]
    model_args, data_args, training_args = parser.parse_json_file(config_file)
    data_args,model_args,training_args = check_args(data_args,model_args,training_args)

    #save config file
    output_dir = training_args.output_dir
    if not os.path.isdir(output_dir):
        os.system(f"mkdir {output_dir}")
    os.system(f"cp {config_file} {output_dir}/run_config.json")
    #os.system(f"copy {config_file} {output_dir}\\run_config.json")

    # Detecting last checkpoint.
    last_checkpoint = None      #初始化 last_checkpoint 变量为 None，用于存储最后一个检查点的路径
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:   # 1. training_args.output_dir 是一个已存在的目录2. training_args.do_train 为 True，表示要进行训练3. training_args.overwrite_output_dir 为 False，表示不覆盖输出目录
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    # 此部分代码主要用于设置日志相关操作，同时将模型和工具代码复制到输出目录'../results/bartlarge/samsum/running'
    os.makedirs(training_args.output_dir,exist_ok=True)
    os.system(f'cp -r model {training_args.output_dir}')
    os.system(f'cp -r utils {training_args.output_dir}')
    # 配置日志记录的基本设置
    logging.basicConfig(
        format="%(asctime)s-%(levelname)s-%(name)s%(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    #logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # 在每个进程上记录简要的训练信息摘要
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # 仅在主进程中将 Transformers 库的日志记录级别设置为 INFO
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()     # 设置 Transformers 库的日志记录级别为 INFO，意味着只记录 INFO 及以上级别的日志信息
    # log args  记录训练所使用的各种参数信息，方便后续查看和调试
    args_dict = {"data_args":data_args,"model_args":model_args,"training_args":training_args}   # 创建一个字典，将数据参数、模型参数和训练参数整合在一起
    keys_ls = list(asdict(training_args).keys()) + list(asdict(model_args).keys()) + list(asdict(data_args).keys())   # asdict 函数将 dataclass 对象转换为字典，方便获取其属性名
    max_width = max([len(arg_name) for arg_name in keys_ls])        # 找出所有参数名称中最长的长度，用于后续格式化输出
    for k,v in args_dict.items():
        logger.info("*"*SCREEN_WIDTH)
        logger.info(k)
        for arg_name,arg_value in asdict(v).items():
            logger.info(f"{arg_name:{max_width}}  {arg_value}")


    # 设置随机数种子
    set_seed(training_args.seed)
    # 调用 model_util 模块中的 get_model_tokenizer 函数，传入模型参数 model_args 和数据参数 data_args，该函数会根据传入的参数返回对应的模型和分词器
    model,tokenizer = model_util.get_model_tokenizer(model_args,data_args)
    if model.config.decoder_start_token_id is None:     # 在一些序列到序列（Seq2Seq）模型中，解码器需要一个特定的起始标记来开始生成输出序列
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    # 对数据集进行预处理，预处理是为了让数据符合模型的输入要求，便于后续的训练、评估和预测
    # 在处理文本数据时，需要将输入文本和目标文本进行分词操作，将文本转换为模型可以处理的词元（tokens）序列
    # 函数会从指定的数据源（如文件、数据库等）中加载数据，并返回一个包含原始数据的数据集对象
    raw_dataset = data_util.get_dataset(data_args,model_args)
    train_dataset,eval_dataset,predict_dataset = data_util.data_preprocessing(raw_dataset,tokenizer,training_args,data_args,model_args) # 调用 data_util 模块中的 data_preprocessing 函数，对原始数据集进行进一步的处理
    model.resize_token_embeddings(len(tokenizer))       # 调整模型的词嵌入层大小，使其与分词器的词汇表大小相匹配
    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id    #根据配置决定标签填充标记的 ID，用于后续的数据整理和损失计算
    # 创建一个自定义的数据整理器（data_collator）实例，该数据整理器用于在序列到序列（Seq2Seq）任务中对输入数据进行整理和填充，使其适合模型的输入格式
    data_collator = MyDataCollatorForSeq2Seq(
        tokenizer,      # 分词器，用于将文本转换为模型可接受的输入格式
        model=model,    # 模型实例，数据整理器可能会根据模型的结构和要求进行一些特定的处理
        label_pad_token_id=label_pad_token_id,       # 标签填充标记的 ID，用于在处理标签时对序列进行填充,-100
        pad_to_multiple_of=8 if training_args.fp16 else None,   # 如果使用混合精度训练（fp16），则将序列长度填充为 8 的倍数，以提高 GPU 计算效率
        dis_offset = int(model.config.max_utt_num-1),       # 距离偏移量，可能用于图相关的特征处理，这里将其设置为模型配置中最大话语数量减 1
        adj_mat_ls = model_args.feature_types           # 邻接矩阵列表，包含需要处理的图特征类型，用于指导数据整理器对图相关特征进行处理,4种图
    )


    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = {}
        metrics_ls = [get_rouge_score]#,get_bert_score,get_meteor_score]
        for metrics in metrics_ls:
            res = metrics(decoded_preds,decoded_labels)
            result.update(res)
        # keys: rouge-1(f,p,r),rouge-2,rouge-l,bert_p,bert_r,bert_f,meteor
        # Extract a few results from ROUGE
        result['rouge-1'] = result['rouge-1']['f'] * 100
        result['rouge-2'] = result['rouge-2']['f'] * 100
        result['rouge-l'] = result['rouge-l']['f'] * 100

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    callbacks_ls = []
    '''if training_args.comet_enable:
        callbacks_ls.append(CometCallBack(training_args))'''
    if training_args.freeze_fine_tune_enable:       # FineTuneCallBack 可能用于实现冻结和微调模型参数的功能
        callbacks_ls.append(FineTuneCallBack(model_args))


    trainer = Seq2SeqTrainer(
        model=model,        # 传入要训练的模型实例
        args=training_args,      # 传入训练相关的参数，如学习率、训练轮数、批次大小等
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,    # 传入分词器，用于处理输入和输出文本
        data_collator=data_collator,    # 传入数据整理器，用于将多个样本整理成一个批次，并进行必要的填充和截断
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,    # 在预测时使用生成模式，传入计算评估指标的函数；否则传入 None
        callbacks = callbacks_ls,   # 传入回调函数列表，这些回调函数会在训练过程的特定阶段执行额外的操作
        #optimizers = (optimizer,scheduler)
    )

    # Training
    if training_args.do_train:
        checkpoint = None       # 初始化检查点变量，用于存储恢复训练的检查点路径
        if training_args.resume_from_checkpoint is not None:    # 检查是否指定了从特定检查点恢复训练
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:   # 如果没有指定特定检查点，但存在最后一个检查点
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)     # 调用 Seq2SeqTrainer 的 train 方法开始训练，传入恢复训练的检查点路径
        trainer.save_model()  # 保存训练好的模型，同时也会保存分词器，方便后续上传

        metrics = train_result.metrics  # 获取训练结果中的指标
        max_train_samples = (           # 确定最大训练样本数，如果指定了最大训练样本数，则使用该值；否则使用训练数据集的长度
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))   # 在指标中添加训练样本数，取最大训练样本数和训练数据集长度的较小值

        trainer.log_metrics("train", metrics)       # 记录训练指标，将指标信息输出到日志中
        trainer.save_metrics("train", metrics)      # 保存训练指标到文件，方便后续分析
        trainer.save_state()                             # 保存训练状态，包括当前的步数、epoch 等信息

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(
            max_length=data_args.val_max_target_length, num_beams=data_args.num_beams, metric_key_prefix="eval"
        )
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            predict_dataset,
            metric_key_prefix="predict",
            max_length=data_args.val_max_target_length,
            num_beams=data_args.num_beams,
        )
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                with open(output_prediction_file, "w",encoding='utf-8') as writer:
                    writer.write("\n".join(predictions))
    
    # 清除所有的checkpoint
    file_ls = os.listdir(training_args.output_dir)
    for file in file_ls:
        if file.startswith('checkpoint'):
            os.system(f'rm -rf {os.path.join(training_args.output_dir,file)}')

    all_results_dir = os.path.join(training_args.output_dir,"all_results.json")
    best_rouge = json.load(open(all_results_dir))["eval_rouge-1"]
    
    log_dir = "/".join(training_args.output_dir.split('/')[:-1])
    
    os.system(f"mv {training_args.output_dir} {os.path.join(log_dir,str(best_rouge))}")
    return results


if __name__ == "__main__":
    main()
