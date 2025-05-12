from torch.utils import data
from transformers import AutoConfig,AutoModel,AutoTokenizer
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
from transformers.models.bart import BartConfig
#from model import bart_model,graphtransformer
from dataclasses import asdict
from src.model import graphtransformer, bart_model

def get_model_tokenizer(model_args,data_args):

    '''
        这个函数的主要功能是根据传入的模型参数 model_args 和数据参数 data_args 来加载预训练的 BART 模型和对应的分词器，并对模型和分词器进行一些必要的配置。具体步骤包括：
        1 加载分词器：使用 AutoTokenizer.from_pretrained 方法从预训练模型中加载分词器。
        2 创建模型配置：根据 model_args 创建图 Transformer 的配置对象。
        3 加载模型：使用 BART.from_pretrained 方法从预训练模型中加载 BART 模型，并应用之前创建的配置。
        4 设置模型输出注意力权重：将模型的配置中的 output_attentions 属性设置为 True，以便在模型运行时输出注意力权重。
        5 添加特殊标记：检查分词器的额外特殊标记中是否包含 <sep> 标记，如果不包含则添加该标记。
        6 返回模型和分词器：将加载好的模型和分词器作为元组返回。
    '''

    tokenizer = AutoTokenizer.from_pretrained(model_args.backbone_model,use_fast=model_args.use_fast_tokenizer)
    config = graphtransformer.GraphTransformerConfig(**asdict(model_args))  # 创建图Transformer配置对象，asdict(model_args) 将 model_args 对象转换为字典，然后使用字典解包的方式传递给 GraphTransformerConfig 类
    
    model = bart_model.BART.from_pretrained(model_args.backbone_model,config=config)
    model.config.output_attentions = True
    if '<sep>' not in tokenizer.additional_special_tokens:
        special_tokens_dict = {"additional_special_tokens":["<sep>"]}#,"#Person2#","#Person1#","#Person3#","#Person4#","#Person5#","#Person6#"]}
        tokenizer.add_special_tokens(special_tokens_dict)
    return model,tokenizer