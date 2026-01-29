
from __future__ import absolute_import, division, print_function
import pandas as pd
import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
from evaluator.evaluator import evaluators
import torch.nn.functional as F
from tokenizers.pre_tokenizers import PreTokenizer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import NormalizedString, PreTokenizedString
from typing import List
from tokenizers import Tokenizer
from tokenizers import normalizers, decoders
from tokenizers.normalizers import StripAccents, unicode_normalizer_from_str, Replace
from tokenizers.processors import TemplateProcessing
from tokenizers import processors, pre_tokenizers
from tokenizers.models import BPE
from attacker import AttackerRandom,Attacker,InsAttacker,Attackertrain,InsAttackertrain,AttackerRandomtrain
from qz3 import Calculated_weight



import numpy as np
import copy
from pathlib import Path
import pandas as pd
import random
import re
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchtext
# import torchtext.vocab as vocab
import sklearn.metrics
from transformers import RobertaModel
from transformers import RobertaConfig
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from torch.autograd import Variable
from torch import nn, optim
from torch.optim import SGD, Adam, RMSprop
from torch.utils.data import Dataset, DataLoader, IterableDataset
from clang import *
import warnings
import json







import numpy as np
import torch
from torch.utils.data import (
    DataLoader,
    Dataset,
    SequentialSampler,
    RandomSampler,
    TensorDataset,
)
from torch.utils.data.distributed import DistributedSampler
import json

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange
import multiprocessing
from model import myCNN

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    get_linear_schedule_with_warmup,
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTConfig,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    DistilBertConfig,
    DistilBertForMaskedLM,
    DistilBertTokenizer,
)


os.environ["TOKENIZERS_PARALLELISM"] = "false"


logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    "openai-gpt": (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "bert": (BertConfig, BertForMaskedLM, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
}


class InputFeatures(object):
    def __init__(
        self,
        input_tokens,
        input_ids,
        idx,
        label,
        uid,
        instab,
        code_tokens
        # weight_V
    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.idx = str(idx)
        self.label = label
        self.uid = uid
        self.instab= instab
        self.code_tokens = code_tokens
        # self.weight_V = weight_V


def convert_examples_to_features(js, my_tokenizer, args,attack_flag):
    # source
    # code = js["func"]
    # pat = re.compile(r'(/\*([^*]|(\*+[^*/]))*\*+/)|(//.*)')
    # code = re.sub(pat,'',Ycode)
    # # code = re.sub('\n','',code)
    # code_tokens = re.sub('\t','',code)
    if js["project"] == "attack_train":
        source_ids = js["func"]
        source_tokens,uid,instab,code_tokens = 0,0,0,0
    else:
        pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
        )
        code = re.sub(pattern, '', js["func"])
        if attack_flag == 1:  
            uid,instab=Calculated_weight(code, my_tokenizer, args)
        else:
            uid,instab=0,0
        
        code_tokens = my_tokenizer.encode(code, add_special_tokens=False).ids
        # ZXC = my_tokenizer.encode(code_tokens,is_pretokenized=True).ids
        source_tokens= code_tokens[: args.block_size - 2]
        source_ids = [0] + source_tokens + [2]
        # source_idss = my_tokenizer.encode(code).ids
        padding_length = args.block_size - len(source_ids)
        source_ids += [1] * padding_length

        # uid=0
        # instab=0
    return InputFeatures(source_tokens, source_ids, js["idx"], js["target"],uid ,instab, code_tokens)


class TextDataset(Dataset):
    def __init__(self, my_tokenizer, args, file_path=None,attack_flag=None):
        self.examples = []
        self.uids = []  # 添加uid列表存储
        self.instab = []
        self.code_tokens = []
        sum =0
        with open(file_path) as f:
            for line in f:
                js = json.loads(line.strip())
                feature = convert_examples_to_features(js, my_tokenizer, args, attack_flag)
                sum =sum +1
                self.examples.append(feature)
                self.uids.append(feature.uid)  # 存储每个样本的uid
                self.instab.append(feature.instab)
                self.code_tokens.append(feature.code_tokens)
                
        # if "train" in file_path:
        #     for idx, example in enumerate(self.examples[:3]):
        #         logger.info("*** Example ***")
        #         logger.info("idx: {}".format(idx))
        #         logger.info("label: {}".format(example.label))
        #         logger.info(
        #             "input_tokens: {}".format(
        #                 [x.replace("\u0120", "_") for x in example.input_tokens]
        #             )
        #         )
        #         logger.info(
        #             "input_ids: {}".format(" ".join(map(str, example.input_ids)))
        #         )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return (
            torch.tensor(self.examples[i].input_ids),
            torch.tensor(self.examples[i].label),
            self.examples[i].idx
        )


class MyTokenizer:
    cidx = cindex.Index.create()

    def clang_split(self, i: int, normalized_string: NormalizedString) -> List[NormalizedString]:
        tok = []
        tu = self.cidx.parse('tmp.c',
                       args=[''],  
                       unsaved_files=[('tmp.c', str(normalized_string.original))],  
                       options=0)
        for t in tu.get_tokens(extent=tu.cursor.extent):
            spelling = t.spelling.strip()
            if spelling == '':
                continue
            tok.append(NormalizedString(spelling))
        return(tok)
    
    def pre_tokenize(self, pretok: PreTokenizedString):
        pretok.split(self.clang_split)

        






def set_seed(seed=2023):
    random.seed(seed)
    os.environ["PYHTONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args, train_dataset, model, my_tokenizer):
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = (
        RandomSampler(train_dataset)
        if args.local_rank == -1
        else DistributedSampler(train_dataset)
    )

    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        num_workers=4,
        pin_memory=True,
    )
    if args.do_attacktrain:    
        attack_flag = 1
        attack_dataset = TextDataset(tokenizer, args, args.Originally_train_data_file, attack_flag)
        attack_flag = 0
    args.max_steps = args.epoch * len(train_dataloader)
    args.save_steps = len(train_dataloader)
    args.warmup_steps = len(train_dataloader)
    args.logging_steps = len(train_dataloader)
    args.num_train_epochs = args.epoch
    model.to(args.device)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.max_steps * 0.12,
        num_training_steps=args.max_steps *1.2,
    )
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level
        )

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    checkpoint_last = os.path.join(args.output_dir, "checkpoint-last")
    scheduler_last = os.path.join(checkpoint_last, "scheduler.pt")
    optimizer_last = os.path.join(checkpoint_last, "optimizer.pt")
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last))
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(torch.load(optimizer_last))
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info(
        "  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size
    )
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)

    global_step = args.start_step
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_mrr = 0.0
    best_acc = 0.0
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()

    for idx in range(args.start_epoch, int(args.num_train_epochs)):
        # bar = tqdm(train_dataloader, total=len(train_dataloader))
        # 检查args.attack_train文件是否存在且不为空
        global_step_d = 0
        if os.path.exists(args.attack_train) and os.path.getsize(args.attack_train) > 0:
            logger.info("***** 检测到对抗样本文件，正在混合训练数据 *****")
            
            # 获取原始训练数据的inputs和labels
            original_inputs = []
            original_labels = []
            
            for batch in train_dataloader:
                original_inputs.extend(batch[0])
                original_labels.extend(batch[1])
            
            logger.info(f"收集了 {len(original_inputs)} 个原始训练样本")
            
            # 读取对抗样本
            attack_inputs = []
            attack_labels = []
            
            try:
                with open(args.attack_train, 'r') as f:
                    for line in f:
                        js = json.loads(line.strip())
                        attack_inputs.append(torch.tensor(js['func']))
                        attack_labels.append(torch.tensor(int(js['target'])))
                
                logger.info(f"读取了 {len(attack_inputs)} 个对抗样本")
                
                # 合并原始数据和对抗样本
                all_inputs = original_inputs + attack_inputs
                all_labels = original_labels + attack_labels
                
                # 将inputs和labels打包为元组列表，以便随机打乱
                combined_data = list(zip(all_inputs, all_labels))
                random.shuffle(combined_data)
                
                # 解包回inputs和labels
                all_inputs, all_labels = zip(*combined_data)
                
                logger.info(f"混合后的数据集大小: {len(all_inputs)}")
                
                # 将数据拆分为批次
                batches = []
                for i in range(0, len(all_inputs), args.train_batch_size):
                    end_idx = min(i + args.train_batch_size, len(all_inputs))
                    batch_inputs = all_inputs[i:end_idx]
                    batch_labels = all_labels[i:end_idx]
                    batches.append((torch.stack(batch_inputs), torch.stack(batch_labels)))
                
                logger.info(f"创建了 {len(batches)} 个批次")
                
                # 使用混合批次进行训练
                bar = tqdm(batches, total=len(batches))
                args.save_steps = len(batches)
                scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=args.max_steps * 0.1,
                    num_training_steps=args.max_steps,
                )                
            except Exception as e:
                logger.error(f"处理对抗样本时出错: {e}")
                # 出错时使用原始训练数据loader
                logger.info("出错回退到使用原始训练数据")
                bar = tqdm(train_dataloader, total=len(train_dataloader))
        else:
            logger.info("未检测到对抗样本文件或文件为空，使用原始训练数据")
            # 直接使用原始训练数据loader
            bar = tqdm(train_dataloader, total=len(train_dataloader))
    
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):
            inputs = batch[0].to(args.device)
            labels = batch[1].to(args.device)
            # weight_V = batch[2].to(args.device)
            model.train()
            loss, logits = model(inputs,labels)
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    amp.master_params(optimizer), args.max_grad_norm
                )
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss
            avg_loss = round(train_loss / tr_num, 5)
            bar.set_description("epoch {} loss {}".format(idx, avg_loss))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                global_step_d += 1
                output_flag = True
                avg_loss = round(
                    np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4
                )
                if (
                    args.local_rank in [-1, 0]
                    and args.logging_steps > 0
                    and global_step_d % args.logging_steps == 0
                ):
                    logging_loss = tr_loss
                    tr_nb = global_step

                if (
                    args.local_rank in [-1, 0]
                    and args.save_steps > 0
                    and global_step_d % args.save_steps == 0
                ):

                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(
                            args, model, my_tokenizer, eval_when_training=True
                        )
                        for key, value in results.items():
                            logger.info("  %s = %s", key, round(value, 4))
                        # Save model checkpoint

                    if results["eval_acc"] > best_acc:
                        best_acc = results["eval_acc"]
                        logger.info("  " + "*" * 20)
                        logger.info("  Best acc:%s", round(best_acc, 4))
                        logger.info("  " + "*" * 20)

                        checkpoint_prefix = "checkpoint-best-acc"
                        output_dir = os.path.join(
                            args.output_dir, "{}".format(checkpoint_prefix)
                        )
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )
                        output_dir = os.path.join(output_dir, "{}".format("model.bin"))
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)
                        test(args, model, my_tokenizer)
                    else:
                        test(args, model, my_tokenizer)
        if args.do_attacktrain:    
            logger.info("***** Epoch %d 结束，开始执行对抗攻击 *****", idx)
            attacker(args, attack_dataset,model, my_tokenizer)
        # test(args, model, tokenizer)


def evaluate(args, model, my_tokenizer, eval_when_training=False):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir
    attack_flag = 0
    eval_dataset = TextDataset(my_tokenizer, args, args.eval_data_file,attack_flag)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = (
        SequentialSampler(eval_dataset)
        if args.local_rank == -1
        else DistributedSampler(eval_dataset)
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size,
        num_workers=4,
        pin_memory=True,
    )

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    labels = []
    for batch in eval_dataloader:
        inputs = batch[0].to(args.device)
        label = batch[1].to(args.device)
        # weight_V = batch[2].to(args.device)
        with torch.no_grad():
            lm_loss, logit = model(inputs,label)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())
        nb_eval_steps += 1
    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    # 定义函数获取概率最大的类别索引
    def getClass(x):
        return x.index(max(x))
    
    # 将预测转换为pandas Series并应用getClass函数
    probs = pd.Series(logits.tolist())
    preds = probs.apply(getClass)
    # 重置索引
    preds.reset_index(drop=True, inplace=True)
    eval_acc = np.mean(labels == preds)
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)

    result = {
        "eval_loss": float(perplexity),
        "eval_acc": round(eval_acc, 4),
    }
    return result


def test(args, model, my_tokenizer):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    attack_flag = 0
    eval_dataset = TextDataset(my_tokenizer, args, args.test_data_file,attack_flag)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = (
        SequentialSampler(eval_dataset)
        if args.local_rank == -1
        else DistributedSampler(eval_dataset)
    )
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size
    )

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    labels = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        inputs = batch[0].to(args.device)
        label = batch[1].to(args.device)
        with torch.no_grad():
            logit = model(inputs)
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())

    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    # 定义函数获取概率最大的类别索引
    def getClass(x):
        return x.index(max(x))
    
    # 将预测转换为pandas Series并应用getClass函数
    probs = pd.Series(logits.tolist())
    preds = probs.apply(getClass)
    # 重置索引
    preds.reset_index(drop=True, inplace=True)
    preds = preds.to_numpy()
    # preds = logits[:, 0] > 0.5
    test_acc=np.mean(labels==preds)

    # 保存为CSV文件
    import csv
    
    # 获取output_dir的最后一个目录名称
    output_dir_parts = args.output_dir.rstrip('/').split('/')
    csv_filename = output_dir_parts[-1] + '.csv'
    csv_path = os.path.join(args.output_dir, csv_filename)
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 写入CSV文件
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['idx', 'label', 'pred'])  # 写入表头
        for example, label, pred in zip(eval_dataset.examples, labels, preds):
            writer.writerow([example.idx, int(label), int(pred)])
    
    print(f"Results saved to {csv_path}")


    with open(os.path.join(args.output_dir, "predictions.txt"), "w") as f:
        for example, pred in zip(eval_dataset.examples, preds):
            if pred:
                f.write(example.idx + "\t1\n")
            else:
                f.write(example.idx + "\t0\n")
                
    result = {
        "test_acc": round(test_acc, 4),
    }
    if result["test_acc"] > 0.63:
        args.test_best = result["test_acc"]
        output_dir = "/root/workspace/zlt/vulberta/models/bast_model"
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )
        filename = "acc_{}_model.bin".format(test_acc)
        output_dirs = os.path.join(output_dir,filename )
        torch.save(model_to_save.state_dict(), output_dirs)
        logger.info("Saving test model to %s", output_dirs)
    scores=evaluators(args)
    print(scores)
    output_dir = "/root/workspace/zlt/vulberta/models"
    with open(os.path.join(output_dir, "log_acc.txt"), "a") as f:
         f.write("{}\n".format(scores))
    return result


def attacker(args, attack_dataset, model, my_tokenizer):
    """
    执行对抗攻击
    """
    logger.info("***** Running attack *****")
    
    # 设置模型为评估模式
    model.eval()
    test_data_dir = os.path.dirname(args.test_data_file)
    attack_flag = 1
    # 加载测试数据
    if args.do_attack:
        eval_dataset = TextDataset(my_tokenizer, args, args.test_data_file,attack_flag)
        uid_test_file = os.path.join(test_data_dir, 'uid_test.jsonl')
        uid_test = set()
        if os.path.exists(uid_test_file):
            with open(uid_test_file, 'r') as f:
                uid_test = {json.loads(line)['uid'] for line in f}
    else:
        train_dataset = TextDataset(my_tokenizer, args, args.train_data_file,attack_flag)
        uid_train_file = os.path.join(test_data_dir, 'uid_train.jsonl')
        # 读取所有uid
        uid_train = set()
        if os.path.exists(uid_train_file):
            with open(uid_train_file, 'r') as f:
                uid_train = {json.loads(line)['uid'] for line in f}
    attack_flag = 0               
    # 使用梯度的词替换
    # attacker = Attacker(model, my_tokenizer, uid_test, args)
    # results = attacker.attack_all(args,eval_dataset, args.n_candidate ,args.n_iter)    
    # # 增加或删除部分语句
    # attacker = InsAttacker(model, my_tokenizer,args)
    # results = attacker.attack_all(args,eval_dataset, args.n_candidate ,args.n_iter)  

    # #使用随机词替换
    attacker = AttackerRandom(model, my_tokenizer, uid_test, args)
    results = attacker.attack_all(args,eval_dataset,args.n_iter)    

   #使用梯度的词替换并保存结果
    # attacker = Attackertrain(model, my_tokenizer, uid_train, args)
    # results = attacker.attack_all(args,train_dataset, args.n_candidate ,args.n_iter)    

    # #增加或删除部分语句
    # attacker = InsAttackertrain(model, my_tokenizer,args)
    # results = attacker.attack_all(args,train_dataset, args.n_candidate ,args.n_iter)  

    # # #使用随机词替换
    # attacker = AttackerRandomtrain(model, my_tokenizer, uid_train, args)
    # results = attacker.attack_all(args,train_dataset,args.n_iter)  

    # 记录攻击结果
    for key, value in results.items():
        logger.info("  %s = %s", key, str(round(value, 5)))
        
    return results


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument(
        "--train_data_file",
        default=None,
        type=str,
        required=True,
        help="The input training data file (a text file).",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    ## Other parameters
    parser.add_argument(
        "--eval_data_file",
        default=None,
        type=str,
        help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
    )
    parser.add_argument(
        "--test_data_file",
        default=None,
        type=str,
        help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
    )

    parser.add_argument(
        "--model_type",
        default="bert",
        type=str,
        help="The model architecture to be fine-tuned.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="The model checkpoint for weights initialization.",
    )

    parser.add_argument(
        "--mlm",
        action="store_true",
        help="Train with masked-language modeling loss instead of language modeling.",
    )
    parser.add_argument(
        "--mlm_probability",
        type=float,
        default=0.15,
        help="Ratio of tokens to mask for masked language modeling loss",
    )

    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Optional pretrained config name or path if not the same as model_name_or_path",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)",
    )
    parser.add_argument(
        "--block_size",
        default=-1,
        type=int,
        help="Optional input sequence length after tokenization."
        "The training dataset will be truncated in block of this size for training."
        "Default to the model max input length for single sentence inputs (take into account special tokens).",
    )
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--do_test", action="store_true", help="Whether to run eval on the dev set."
    )

    parser.add_argument(
        "--do_attacktrain", 
        action="store_true",
        help="Whether to run attack on the test set.",
    )

    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--train_batch_size",
        default=4,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--eval_batch_size",
        default=4,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight deay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=1.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps."
    )

    parser.add_argument(
        "--logging_steps", type=int, default=50, help="Log every X updates steps."
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=50,
        help="Save checkpoint every X updates steps.",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Avoid using CUDA when available"
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--epoch", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--server_ip", type=str, default="", help="For distant debugging."
    )
    parser.add_argument(
        "--server_port", type=str, default="", help="For distant debugging."
    )
    
    # 添加攻击相关的参数
    parser.add_argument(
        "--do_attack", 
        action="store_true",
        help="Whether to run attack on the test set."
    )
    parser.add_argument(
        "--attack_type",
        type=str,
        default="token_random",
        choices=["token", "insert", "token_random", "insert_random"],
        help="Type of attack to perform"
    )
    parser.add_argument(
        "--n_candidate", 
        type=int,
        default=5,
        help="Number of candidates for attack"
    )
    parser.add_argument(
        "--n_iter",
        type=int, 
        default=20,
        help="Maximum number of iterations for attack"
    )
    parser.add_argument(
        "--EMBED_SIZE",
        type=int, 
        default=5000,
        help="vulberta 参数"
    )
    parser.add_argument(
        "--EMBED_DIM",
        type=int, 
        default=768,
        help="vulberta 参数"
    )
    parser.add_argument(
        "--attack_train",
        default="/root/workspace/zlt/Ygraphcodebert/dataset/attack.jsonl",
        type=str,
        help="保存对抗数据",
    )
    parser.add_argument(
        "--Originally_train_data_file",
        default="/root/workspace/zlt/Ygraphcodebert/dataset/train.jsonl",
        type=str,
        help="保存原始数据",
    )
    args = parser.parse_args()

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(
            address=(args.server_ip, args.server_port), redirect_output=True
        )
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    args.per_gpu_train_batch_size = args.train_batch_size // args.n_gpu
    args.per_gpu_eval_batch_size = args.eval_batch_size // args.n_gpu
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s)",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args.seed)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    args.start_epoch = 0
    args.start_step = 0
    checkpoint_last = os.path.join(args.output_dir, "checkpoint-last")
    if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        args.model_name_or_path = os.path.join(checkpoint_last, "pytorch_model.bin")
        args.config_name = os.path.join(checkpoint_last, "config.json")
        idx_file = os.path.join(checkpoint_last, "idx_file.txt")
        with open(idx_file, encoding="utf-8") as idxf:
            args.start_epoch = int(idxf.readlines()[0].strip()) + 1

        step_file = os.path.join(checkpoint_last, "step_file.txt")
        if os.path.exists(step_file):
            with open(step_file, encoding="utf-8") as stepf:
                args.start_step = int(stepf.readlines()[0].strip())

        logger.info(
            "reload model from {}, resume from {} epoch".format(
                checkpoint_last, args.start_epoch
            )
        )

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    # config_class, model_class= MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    config.num_labels = 1
    # tokenizer = tokenizer_class.from_pretrained(
    #     args.tokenizer_name,
    #     do_lower_case=args.do_lower_case,
    #     cache_dir=args.cache_dir if args.cache_dir else None,
    # )
    # 从词汇表和合并文件加载预训练的分词器
    # Load pre-trained tokenizers from vocabulary and merges files
    vocab, merges = BPE.read_file(vocab="/root/workspace/zlt/vulberta/tokenizer/drapgh-vocab.json", merges="/root/workspace/zlt/vulberta/tokenizer/drapgh-merges.txt")
    # 使用词汇表和合并规则初始化BPE分词器，设置未知标记为"<unk>"
    # Initialize BPE tokenizer with vocabulary and merges, setting unknown token to "<unk>"
    my_tokenizer = Tokenizer(BPE(vocab, merges, unk_token="<unk>"))

    # 配置分词器的标准化器，去除重音符号并将空格替换为"Ä"
    # Configure tokenizer normalizer to strip accents and replace spaces with "Ä"
    my_tokenizer.normalizer = normalizers.Sequence([StripAccents(), Replace(" ", "Ä")])
    # 设置自定义的预分词器，使用MyTokenizer类
    # Set custom pre-tokenizer using MyTokenizer class
    my_tokenizer.pre_tokenizer = PreTokenizer.custom(MyTokenizer())
    # 配置字节级后处理，不修剪偏移量
    # Configure byte-level post-processing without trimming offsets
    my_tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    # 设置模板处理，在序列周围添加特殊标记
    # Set template processing to add special tokens around sequences
    my_tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",  # 模板格式：开始标记，序列，结束标记
                            # Template format: start token, sequence, end token
        special_tokens=[       # 定义特殊标记及其ID
                            # Define special tokens and their IDs
        ("<s>",0),    # 开始标记 Start token
        ("<pad>",1),  # 填充标记 Padding token  
        ("</s>",2),   # 结束标记 End token
        ("<unk>",3),  # 未知标记 Unknown token
        ("<mask>",4)  # 掩码标记 Mask token
        ]
    )

    my_tokenizer.enable_truncation(max_length=1024)
    # my_tokenizer.enable_padding(direction='right', pad_id=1, pad_type_id=0, pad_token='<pad>', length=1024, pad_to_multiple_of=None)
 
    # tokenizer = my_tokenizer
    if args.block_size <= 0:
        args.block_size = (
            1024
        )  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, 1024)
    # if args.model_name_or_path:
    #     model = model_class.from_pretrained(
    #         args.model_name_or_path,
    #         from_tf=bool(".ckpt" in args.model_name_or_path),
    #         config=config,
    #         cache_dir=args.cache_dir if args.cache_dir else None,
    #     )
    # else:
    #     model = model_class(config)
    model = RobertaModel.from_pretrained('/root/workspace/zlt/vulberta/models/VulBERTa/').embeddings.word_embeddings.weight



    PAD_IDX = 1
    model = myCNN(model, args.EMBED_SIZE, args.EMBED_DIM)
    model.embed.weight.data[PAD_IDX] = torch.zeros(args.EMBED_DIM)
    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache
        attack_flag = 0
        train_dataset = TextDataset(my_tokenizer, args, args.train_data_file,attack_flag)
        if args.local_rank == 0:
            torch.distributed.barrier()

        # beat_model_bin="/root/workspace/zlt/Ygraphcodebert/code/saved_models/checkpoint-best-acc/model.bin"
        # if os.path.isfile(beat_model_bin):      #判断保存的最好的模型数据在不在，这样可以接着训练，如何想从头训练把这两行注释掉。
        #     model.load_state_dict(torch.load(beat_model_bin))

        train(args, train_dataset, model, my_tokenizer)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoint_prefix = "checkpoint-best-acc/model.bin"
        output_dir = os.path.join(args.output_dir, "{}".format(checkpoint_prefix))
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        result = evaluate(args, model, my_tokenizer)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key], 4)))

    if args.do_test and args.local_rank in [-1, 0]:
        checkpoint_prefix = "checkpoint-best-acc/model.bin"
        output_dir = os.path.join(args.output_dir, "{}".format(checkpoint_prefix))
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        test(args, model, my_tokenizer)

    # Attack
    if args.do_attack:
        logger.info("***** Running attack *****")
        logger.info("  Attack type = %s", args.attack_type)
        logger.info("  Number of candidates = %d", args.n_candidate)
        logger.info("  Number of iterations = %d", args.n_iter)
        
        # 加载最佳模型
        checkpoint_prefix = "checkpoint-best-acc/model.bin"
        output_dir = os.path.join(args.output_dir, "{}".format(checkpoint_prefix))
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        
        attack_flag = 1
        attack_dataset = TextDataset(tokenizer, args, args.test_data_file, attack_flag)
        attack_flag = 0
        # 执行攻击
        attack_results = attacker(args, attack_dataset,model, my_tokenizer)
        
        # 记录攻击结果
        for key, value in attack_results.items():
            logger.info("  %s = %s", key, str(round(value, 5)))

    return results


if __name__ == "__main__":
    main()
