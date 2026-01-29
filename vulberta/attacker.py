# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 15:26:02 2020

@author: DrLC
"""

from modifier import TokenModifier, InsModifier


import numpy
import random
import torch
import torch.nn as nn
import argparse
import pickle, gzip
import os, sys, time
from evaluator.evaluator import calculate_scores
from sklearn import metrics
from torch.utils.data import DataLoader, SequentialSampler, DistributedSampler
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score
import pandas as pd
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import json

class Attacker(object):
    
    def __init__(self, model, my_tokenizer, uids,args):
        self.model = model
        self.my_tokenizer = my_tokenizer
        self.tokenM = TokenModifier(
            classifier=model,
            loss=torch.nn.CrossEntropyLoss(),
            uids=uids,
            my_tokenizer=my_tokenizer,
            args = args
        )
        self.args = args
        
    
    def attack(self, source_ids, label, uids, n_candidate=5, n_iter=20):
        """
        执行随机攻击
        Args:
            source_ids: 输入序列的token ids
            label: 真实标签
            uids: 可替换的标识符信息
            n_iter: 最大攻击迭代次数
        """
        iter = 0
        n_stop = 0
        
        # 获取原始预测
        with torch.no_grad():
            logit = self.model(source_ids)
            def getClass(x):
                return x.index(max(x))
            
            # 将预测转换为pandas Series并应用getClass函数
            probs = pd.Series(logit.tolist())
            preds = probs.apply(getClass)
            # 重置索引
            preds.reset_index(drop=True, inplace=True)
            
            # 将 preds 转换为 numpy 数组
            preds_numpy = preds.to_numpy()

            # 获取 label 所在的设备
            device = label.device

            # 将 numpy 数组转换为相同设备上的 tensor
            preds_tensor = torch.tensor(preds_numpy, device=device)
            old_pred = preds_tensor
            old_prob = logit
            old_prob, _ = torch.max(old_prob, dim=1)
            
        # labels = label.item()
        if old_pred != label:
            # print(old_pred, label, "居然不相等")
            return True, old_pred
            
        while iter < n_iter:
            # 获取所有可替换的标识符
            keys = list(uids.keys())
            made_changes = False  # 标记是否在本轮中做了有效替换
            
            for k in keys:
                if iter >= n_iter:
                    break
                if n_stop >= len(uids):
                    iter = n_iter
                    break
                
                # 这部分不需要了，因为我已经提前筛选了
                # Gk = 'Ġ' + k
                Gk_idx = self.my_tokenizer.encode(k).ids[1]
                if Gk_idx == 3:
                    continue

                iter += 1
                # 随机替换标识符
                new_source_ids, new_uid = self.tokenM.rename_uid(source_ids, label, k, n_candidate)
                
                if new_source_ids is None:
                    n_stop += 1
                    # print(f"skip uid\t{k}")
                    continue
                    
                # 将new_source_ids堆叠为批处理格式
                # new_source_ids = torch.stack(new_source_ids)
                # 将new_source_ids堆叠为批处理格式并移除多余的维度
                new_source_ids = torch.stack(new_source_ids).squeeze(1)  # [n_candidate, block_size]
                # 获取新预测
                with torch.no_grad():
                    outputs = self.model(new_source_ids)  # [batch_size]
                    def getClass(x):
                        return x.index(max(x))
                    
                    # 将预测转换为pandas Series并应用getClass函数
                    probs = pd.Series(outputs.tolist())
                    preds = probs.apply(getClass)
                    # 重置索引
                    preds.reset_index(drop=True, inplace=True)
                    
                    # 将 preds 转换为 numpy 数组
                    preds_numpy = preds.to_numpy()

                    # 获取 label 所在的设备
                    device = label.device

                    # 将 numpy 数组转换为相同设备上的 tensor
                    preds_tensor = torch.tensor(preds_numpy, device=device)
                    new_preds = preds_tensor

                    # 获取outputs中每行的最大值，保持维度
                    new_probs, _ = torch.max(outputs, dim=1)  # 形状变为[5]，每个元素是对应行的最大值

                    # 如果需要保持[5,1]的形状，可以使用unsqueeze
                    new_probs = new_probs.unsqueeze(1)  # 形状变为[5,1]

                    # 检查是否有攻击成功的样本
                    for uid, pred, prob in zip(new_uid, new_preds, new_probs):
                        if pred != label:
                            print(f"SUCC!\t{k} => {uid}\t\t{label.item()}({old_prob.item():.5f}) => {pred.item()}({prob.item():.5f})")
                            return True, pred

                    # 如果没有成功的攻击，选择最佳候选项（概率最小的）
                    best_idx = torch.argmin(new_probs)
                    if new_probs[best_idx] < old_prob:
                        source_ids = new_source_ids[best_idx].unsqueeze(0)  # 添加unsqueeze(0)为了维持形状为[1, block_size]，不然就只是[block_size]
                        uids[new_uid[best_idx]] = uids.pop(k)
                        n_stop = 0
                        made_changes = True  # 标记本轮做了有效替换
                        print(f"acc\t{k} => {new_uid[best_idx]}\t\t{label.item()}({old_prob.item():.5f}) => {label.item()}({new_probs[best_idx].item():.5f})")
                        old_prob = new_probs[best_idx]
                    else:
                        n_stop += 1
                        # print(f"rej\t{k}")
            
            # 如果本轮中没有做任何有效替换，则退出循环
            if not made_changes:
                break
        
        # print("FAIL!")
        return False, label

    def attack_all(self,args,eval_dataset, n_candidate=100,n_iter=20):
        """
        对整个数据集进行攻击
        """
        n_succ = 0
        total_time = 0
        trues = []
        preds = []

        # 创建列表保存idx, label, pred数据
        idx_list = []
        label_list = []
        pred_list = []

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset,
            sampler=eval_sampler, 
            batch_size=1
        )

        logits = []
        labels = []
        
        # 遍历时使用索引
        for i, batch in enumerate(tqdm(eval_dataloader, total=len(eval_dataloader))):
            inputs = batch[0].to(args.device)
            label = batch[1].to(args.device)
            idx = batch[2]
            current_uid = eval_dataset.uids[i]  # 使用索引获取对应的uid
            
            start_time = time.time()
            success, pred = self.attack(inputs, label,current_uid, n_candidate,n_iter)
            # i = i + 1 #添加一个计数器

            # 保存当前样本的idx, label和pred
            idx_list.append(str(idx))
            label_list.append(label.item())
            pred_list.append(pred.item() if torch.is_tensor(pred) else pred)

            
            if success:
                n_succ += 1
                # total_time += time.time() - start_time
                
            preds.append(pred)
            trues.append(label)
            
            # 打印当前结果
            # curr_acc = n_succ/(i+1)
            # avg_time = total_time/n_succ if n_succ > 0 else float('nan')
            # print(f"\tCurr succ rate = {curr_acc:.3f}, Avg time cost = {avg_time:.1f} sec")
            
            # 计算评估指标
        # precision = metrics.precision_score(trues, preds, average='macro')
        # recall = metrics.recall_score(trues, preds, average='macro')
        # f1 = metrics.f1_score(trues, preds, average='macro')
        # print(f"\t(P, R, F1) = ({precision:.3f}, {recall:.3f}, {f1:.3f})")
        if not torch.is_tensor(trues):
            trues = torch.tensor(trues)
        if not torch.is_tensor(preds):
            preds = torch.tensor(preds)
            
        # 如果在GPU上，移到CPU
        if trues.is_cuda:
            trues = trues.cpu()
        if preds.is_cuda:
            preds = preds.cpu()

        # 计算单个指标
        # 计算准确率
        acc = accuracy_score(trues, preds)
        precision = precision_score(trues, preds)
        recall = recall_score(trues, preds)
        f1 = f1_score(trues, preds)

        # 保存数据到CSV文件
        import csv
        import os
        
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
            for idx, label, pred in zip(idx_list, label_list, pred_list):
                writer.writerow([idx, label, pred])
        
        print(f"Results saved to {csv_path}")        
        
        return {
            "success_rate": n_succ/len(eval_dataset),
            # "avg_time": total_time/n_succ if n_succ > 0 else float('nan'),
            "acc":acc,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        


class InsAttacker(object):
    
    def __init__(self, model, my_tokenizer,args):
        self.model = model
        self.my_tokenizer = my_tokenizer
        # self.inss = instab
        self.args = args
        self.insM = InsModifier(classifier=model,
                                my_tokenizer = my_tokenizer,
                                poses=None) # wait to init when attack
        # 初始化线程池
        self.num_cores = multiprocessing.cpu_count()
        self.executor = ThreadPoolExecutor(max_workers=self.num_cores)
        
    def Shape_processing(self,code_tokens):
        source_tokens= code_tokens[: self.args.block_size - 2]
        
        # Add start and end tokens
        source_ids = [0] + source_tokens + [2]

        # Calculate padding length
        padding_length = self.args.block_size - len(source_ids)
        source_ids += [1] * padding_length
        return source_ids         

    def Shape_processing_batch(self, new_code_tokens):
        # 使用线程池并行处理
        new_source_ids = list(self.executor.map(self.Shape_processing, new_code_tokens))
        return new_source_ids

    def attack(self,code_tokens, label, poses, n_candidate=5,n_iter=20):
        self.insM.initInsertDict(poses)
        iter = 0  # 初始化迭代次数
        n_stop = 0  # 初始化停止计数器
        #原始数据处理
        device = label.device
        source_ids = self.Shape_processing(code_tokens)
        source_ids = torch.tensor(source_ids).unsqueeze(0).to(device)
        # 获取原始预测
        with torch.no_grad():
            logit = self.model(source_ids)
            def getClass(x):
                return x.index(max(x))
            
            # 将预测转换为pandas Series并应用getClass函数
            probs = pd.Series(logit.tolist())
            preds = probs.apply(getClass)
            # 重置索引
            preds.reset_index(drop=True, inplace=True)
            
            # 将 preds 转换为 numpy 数组
            preds_numpy = preds.to_numpy()

            # 获取 label 所在的设备
            device = label.device

            # 将 numpy 数组转换为相同设备上的 tensor
            preds_tensor = torch.tensor(preds_numpy, device=device)
            old_pred = preds_tensor
            old_prob = logit
            old_prob, _ = torch.max(old_prob, dim=1)

        if old_pred != label:
            # print("SUCC! Original mistake.")
            return True, old_pred

        while iter < n_iter:
        
            iter += 1
            # 随机替换标识符

            # 获取插入和删除候选的数量
            # n_could_del = self.insM.insertDict["count"]
            # n_candidate_del = n_could_del  # 删除候选的数量
            # n_candidate_ins = n_candidate - n_candidate_del  # 插入候选的数量
            # assert n_candidate_del >= 0 and n_candidate_ins >= 0  # 确保候选数量非负
            n_candidate_ins = n_candidate          
            # 生成删除和插入后的新输入数据
            # new_code_tokens_del, new_insertDict_del = self.insM.remove(code_tokens, n_candidate_del)
            new_code_tokens_add, new_insertDict_add = self.insM.insert(code_tokens, n_candidate_ins)
            # new_code_tokens = new_code_tokens_del + new_code_tokens_add  # 合并删除和插入后的新输入数据
            # new_insertDict = new_insertDict_del + new_insertDict_add  # 合并插入字典
            new_code_tokens = new_code_tokens_add
            new_insertDict = new_insertDict_add
            
            if new_code_tokens == []:  # 如果没有有效的候选数据
                n_stop += 1  # 增加停止计数器
                continue  # 跳过本次迭代
            # new_source_ids = []
            # for new_code_token in new_code_tokens:
            #     source_ids = self.Shape_processing(new_code_token)
            #     new_source_ids.append(source_ids)
            # 使用并行处理替换原来的循环
            new_source_ids = self.Shape_processing_batch(new_code_tokens)
            
            # 将列表转换为tensor
            new_source_ids = torch.tensor(new_source_ids).to(device)
            with torch.no_grad():
                logit = self.model(new_source_ids)  # [batch_size]
                def getClass(x):
                    return x.index(max(x))
                
                # 将预测转换为pandas Series并应用getClass函数
                probs = pd.Series(logit.tolist())
                preds = probs.apply(getClass)
                # 重置索引
                preds.reset_index(drop=True, inplace=True)
                
                # 将 preds 转换为 numpy 数组
                preds_numpy = preds.to_numpy()

                # 获取 label 所在的设备
                device = label.device

                # 将 numpy 数组转换为相同设备上的 tensor
                preds_tensor = torch.tensor(preds_numpy, device=device)
                new_preds = preds_tensor
                new_probs = logit
                new_probs, _ = torch.max(new_probs, dim=1)

                # 检查是否有攻击成功的样本
                for insD, p, pr in zip(new_insertDict, new_preds, new_probs):
                    if p != label:
                        print(f"[SUCC] insert_n: {self.insM.insertDict['count']} -> {insD['count']} | label: {label.item()} | prob: {old_prob.item():.3f} -> {pr.item():.3f} | pred: {p.item()}") # 打印成功信息
                        return True, p

                # 如果没有成功的攻击，选择最佳候选项（概率最小的）
                best_idx = torch.argmin(new_probs)
                if new_probs[best_idx] < old_prob:
                    # source_ids = new_source_ids[best_idx].unsqueeze(0)  # 添加unsqueeze(0)为了维持形状为[1, block_size]，不然就只是[block_size]
                    # uids[new_uid[best_idx]] = uids.pop(k)
                    # source_ids = new_source_ids[best_idx]
                    self.insM.insertDict = new_insertDict[best_idx]
                    n_stop = 0
                    print(f"[ACC] insert_n: {self.insM.insertDict['count']} -> {new_insertDict[best_idx]['count']} | label: {label.item()} | prob: {old_prob.item():.3f} -> {new_probs[best_idx].item():.3f} | pred: {p.item()}") # 打印成功信息
                    old_prob = new_probs[best_idx]
                else:
                    n_stop += 1
                    # print(f"rej\t{k}") 
                if n_stop >= len(new_source_ids):    # 如果停止计数器大于等于新输入数据的数量
                    iter = n_iter  # 设置迭代次数为最大迭代次数
                    break  # 跳出循环      
        # print("FAIL!")
        return False,label

    def attack_all(self,args,eval_dataset, n_candidate=100,n_iter=20):
        """
        对整个数据集进行攻击
        """
        n_succ = 0
        total_time = 0
        trues = []
        preds = []
        
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset,
            sampler=eval_sampler, 
            batch_size=1
        )
        label_list = []
        pred_list = []
        logits = []
        labels = []
        
        # 遍历时使用索引
        for i, batch in enumerate(tqdm(eval_dataloader, total=len(eval_dataloader))):
            # inputs = batch[0].to(args.device)
            label = batch[1].to(args.device)
            # current_uid = eval_dataset.uids[i]  # 使用索引获取对应的uid
            current_instab = eval_dataset.instab[i]
            current_code_tokens = eval_dataset.code_tokens[i]
            
            # start_time = time.time()
            success, pred = self.attack(current_code_tokens, label,current_instab, n_candidate,n_iter)
            # 保存当前样本的idx, label和pred
            # idx_list.append(str(idx))
            label_list.append(label.item())
            pred_list.append(pred.item() if torch.is_tensor(pred) else pred)
            
            if success:
                n_succ += 1
                # total_time += time.time() - start_time
                
            preds.append(pred)
            trues.append(label)
            
            # 打印当前结果
            # curr_acc = n_succ/(i+1)
            # avg_time = total_time/n_succ if n_succ > 0 else float('nan')
            # print(f"\tCurr succ rate = {curr_acc:.3f}, Avg time cost = {avg_time:.1f} sec")
            
            # 计算评估指标
        # precision = metrics.precision_score(trues, preds, average='macro')
        # recall = metrics.recall_score(trues, preds, average='macro')
        # f1 = metrics.f1_score(trues, preds, average='macro')
        # print(f"\t(P, R, F1) = ({precision:.3f}, {recall:.3f}, {f1:.3f})")
        if not torch.is_tensor(trues):
            trues = torch.tensor(trues)
        if not torch.is_tensor(preds):
            preds = torch.tensor(preds)
            
        # 如果在GPU上，移到CPU
        if trues.is_cuda:
            trues = trues.cpu()
        if preds.is_cuda:
            preds = preds.cpu()

        # 计算单个指标
        # 计算准确率
        acc = accuracy_score(trues, preds)
        precision = precision_score(trues, preds)
        recall = recall_score(trues, preds)
        f1 = f1_score(trues, preds)

        # 保存数据到CSV文件
        import csv
        import os
        
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
            for examples, label, pred in zip(eval_dataset.examples, label_list, pred_list):
                writer.writerow([examples.idx, label, pred])        
        
        return {
            "success_rate": n_succ/len(eval_dataset),
            # "avg_time": total_time/n_succ if n_succ > 0 else float('nan'),
            "acc":acc,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    def __del__(self):
        # 确保线程池被正确关闭
        self.executor.shutdown(wait=True)

 
class AttackerRandom(object):
    
    def __init__(self, model, my_tokenizer, uids,args):
        self.model = model
        self.my_tokenizer = my_tokenizer
        self.tokenM = TokenModifier(
            classifier=model,
            loss=torch.nn.CrossEntropyLoss(),
            uids=uids,
            my_tokenizer=my_tokenizer,
            args = args
        )
        self.args = args
    
    def attack(self, source_ids, label, uids, n_iter=20):
        """
        执行随机攻击
        Args:
            source_ids: 输入序列的token ids
            label: 真实标签
            uids: 可替换的标识符信息
            n_iter: 最大攻击迭代次数
        """
        iter = 0
        n_stop = 0
        
        # 获取原始预测
        with torch.no_grad():
            logit = self.model(source_ids)
            def getClass(x):
                return x.index(max(x))
            
            # 将预测转换为pandas Series并应用getClass函数
            probs = pd.Series(logit.tolist())
            preds = probs.apply(getClass)
            # 重置索引
            preds.reset_index(drop=True, inplace=True)
            
            # 将 preds 转换为 numpy 数组
            preds_numpy = preds.to_numpy()

            # 获取 label 所在的设备
            device = label.device

            # 将 numpy 数组转换为相同设备上的 tensor
            preds_tensor = torch.tensor(preds_numpy, device=device)
            old_pred = preds_tensor
            old_prob = logit
            old_prob, _ = torch.max(old_prob, dim=1)
            
        # labels = label.item()
        if old_pred != label:
            print(old_pred, label, "居然不相等")
            return True, old_pred
            
        while iter < n_iter:
            # 获取所有可替换的标识符
            keys = list(uids.keys())
            made_changes = False  # 标记是否在本轮中做了有效替换
            for k in keys:
                if iter >= n_iter:
                    break
                if n_stop >= len(uids):
                    iter = n_iter
                    break
                    
                iter += 1
                # 随机替换标识符
                new_source_ids, new_uid = self.tokenM.rename_uid_random(source_ids, k, keys)
                
                if new_source_ids is None:
                    n_stop += 1
                    print(f"skip uid\t{k}")
                    continue
                    
                # 获取新预测
                with torch.no_grad():
                    outputs = self.model(new_source_ids)
                    def getClass(x):
                        return x.index(max(x))
                    
                    # 将预测转换为pandas Series并应用getClass函数
                    probs = pd.Series(outputs.tolist())
                    preds = probs.apply(getClass)
                    # 重置索引
                    preds.reset_index(drop=True, inplace=True)
                    
                    # 将 preds 转换为 numpy 数组
                    preds_numpy = preds.to_numpy()

                    # 获取 label 所在的设备
                    device = label.device

                    # 将 numpy 数组转换为相同设备上的 tensor
                    preds_tensor = torch.tensor(preds_numpy, device=device)
                    new_preds = preds_tensor

                    # 获取outputs中每行的最大值，保持维度
                    new_probs, _ = torch.max(outputs, dim=1)  # 形状变为[5]，每个元素是对应行的最大值

                    # 如果需要保持[5,1]的形状，可以使用unsqueeze
                    new_probs = new_probs.unsqueeze(1)  # 形状变为[5,1]
                
                    # 检查是否有攻击成功的样本
                    # for uid, pred, prob in zip(new_uid, new_preds, new_probs):
                    if new_preds != label:
                        print(f"SUCC!\t{k} => {new_uid}\t\t{label.item()}({old_prob.item():.5f}) => {new_preds.item()}({new_preds.item():.5f})")
                        return True, new_preds

                    # 如果没有成功的攻击，选择最佳候选项（概率最小的）
                    # best_idx = torch.argmin(new_probs)
                    if new_probs < old_prob:
                        source_ids = new_source_ids # 添加unsqueeze(0)为了维持形状为[1, block_size]，不然就只是[block_size]
                        uids[new_uid] = uids.pop(k)
                        n_stop = 0
                        made_changes = True  # 标记本轮做了有效替换
                        print(f"acc\t{k} => {new_uid}\t\t{label.item()}({old_prob.item():.5f}) => {label.item()}({new_probs.item():.5f})")
                        old_prob = new_probs
                    else:
                        n_stop += 1
                        # print(f"rej\t{k}")
            
            # 如果本轮中没有做任何有效替换，则退出循环
            if not made_changes:
                break
        return False, label

    def attack_all(self,args,eval_dataset, n_iter=20):
        """
        对整个数据集进行攻击
        """
        n_succ = 0
        total_time = 0
        trues = []
        preds = []
        
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset,
            sampler=eval_sampler, 
            batch_size=1
        )
        label_list = []
        pred_list = []
        logits = []
        labels = []
        
        # 遍历时使用索引
        for i, batch in enumerate(tqdm(eval_dataloader, total=len(eval_dataloader))):
            inputs = batch[0].to(args.device)
            label = batch[1].to(args.device)
            current_uid = eval_dataset.uids[i]  # 使用索引获取对应的uid
            
            start_time = time.time()
            success, pred = self.attack(inputs, label,current_uid, n_iter)
            label_list.append(label.item())
            pred_list.append(pred.item() if torch.is_tensor(pred) else pred)
            
            if success:
                n_succ += 1
                total_time += time.time() - start_time
                
            preds.append(pred)
            trues.append(label)
            
            # 打印当前结果
            curr_acc = n_succ/(i+1)
            avg_time = total_time/n_succ if n_succ > 0 else float('nan')
            print(f"\tCurr succ rate = {curr_acc:.3f}, Avg time cost = {avg_time:.1f} sec")
            
            # 计算评估指标
        # precision = metrics.precision_score(trues, preds, average='macro')
        # recall = metrics.recall_score(trues, preds, average='macro')
        # f1 = metrics.f1_score(trues, preds, average='macro')
        # print(f"\t(P, R, F1) = ({precision:.3f}, {recall:.3f}, {f1:.3f})")
        if not torch.is_tensor(trues):
            trues = torch.tensor(trues)
        if not torch.is_tensor(preds):
            preds = torch.tensor(preds)
            
        # 如果在GPU上，移到CPU
        if trues.is_cuda:
            trues = trues.cpu()
        if preds.is_cuda:
            preds = preds.cpu()
        # score = calculate_scores(trues, preds)
        

        # 计算单个指标
        # 计算准确率
        acc = accuracy_score(trues, preds)
        precision = precision_score(trues, preds)
        recall = recall_score(trues, preds)
        f1 = f1_score(trues, preds)

        # 保存数据到CSV文件
        import csv
        import os
        
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
            for examples, label, pred in zip(eval_dataset.examples, label_list, pred_list):
                writer.writerow([examples.idx, label, pred])
        
        print(f"Results saved to {csv_path}")         
        
        return {
            "success_rate": n_succ/len(eval_dataset),
            "avg_time": total_time/n_succ if n_succ > 0 else float('nan'),
            "acc":acc,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

class Attackertrain(object):
    
    def __init__(self, model, my_tokenizer, uids, args):
        self.model = model
        self.my_tokenizer = my_tokenizer
        self.tokenM = TokenModifier(
            classifier=model,
            loss=torch.nn.CrossEntropyLoss(),
            uids=uids,
            my_tokenizer=my_tokenizer,
            args = args
        )
        self.args = args
        
        # 初始化攻击样本计数器
        self.attack_sample_counter = 30000
        
        # 确保attack_train文件所在目录存在
        if hasattr(args, 'attack_train') and args.attack_train:
            attack_train_dir = os.path.dirname(args.attack_train)
            if attack_train_dir and not os.path.exists(attack_train_dir):
                os.makedirs(attack_train_dir, exist_ok=True)
    
    def attack(self, source_ids, label, uids, n_candidate=5, n_iter=20):
        """
        执行随机攻击
        Args:
            source_ids: 输入序列的token ids
            label: 真实标签
            uids: 可替换的标识符信息
            n_iter: 最大攻击迭代次数
        """
        iter = 0
        n_stop = 0
        
        # 检查uids是否为有效的字典
        if not isinstance(uids, dict) or len(uids) == 0:
            return False, label, 0
        
        # 获取原始预测
        with torch.no_grad():
            logit = self.model(source_ids)
            def getClass(x):
                return x.index(max(x))
            
            # 将预测转换为pandas Series并应用getClass函数
            probs = pd.Series(logit.tolist())
            preds = probs.apply(getClass)
            # 重置索引
            preds.reset_index(drop=True, inplace=True)
            
            # 将 preds 转换为 numpy 数组
            preds_numpy = preds.to_numpy()

            # 获取 label 所在的设备
            device = label.device

            # 将 numpy 数组转换为相同设备上的 tensor
            preds_tensor = torch.tensor(preds_numpy, device=device)
            old_pred = preds_tensor
            old_prob = logit
            old_prob, _ = torch.max(old_prob, dim=1)
            
        if old_pred != label:
            return True, old_pred, 0
            
        while iter < n_iter:
            # 获取所有可替换的标识符
            keys = list(uids.keys())
            made_changes = False  # 标记是否在本轮中做了有效替换
            
            for k in keys:
                if iter >= n_iter:
                    break
                if n_stop >= len(uids):
                    iter = n_iter
                    break
                
                Gk_idx = self.my_tokenizer.encode(k).ids[1]
                if Gk_idx == 3:
                    continue

                iter += 1
                # 随机替换标识符
                new_source_ids, new_uid = self.tokenM.rename_uid(source_ids, label, k, n_candidate)
                
                if new_source_ids is None:
                    n_stop += 1
                    continue
                    
                new_source_ids = torch.stack(new_source_ids).squeeze(1)  # [n_candidate, block_size]
                # 获取新预测
                with torch.no_grad():
                    outputs = self.model(new_source_ids)  # [batch_size]
                    def getClass(x):
                        return x.index(max(x))
                    
                    # 将预测转换为pandas Series并应用getClass函数
                    probs = pd.Series(outputs.tolist())
                    preds = probs.apply(getClass)
                    # 重置索引
                    preds.reset_index(drop=True, inplace=True)
                    
                    # 将 preds 转换为 numpy 数组
                    preds_numpy = preds.to_numpy()

                    # 获取 label 所在的设备
                    device = label.device

                    # 将 numpy 数组转换为相同设备上的 tensor
                    preds_tensor = torch.tensor(preds_numpy, device=device)
                    new_preds = preds_tensor

                    # 获取outputs中每行的最大值，保持维度
                    new_probs, _ = torch.max(outputs, dim=1)  # 形状变为[5]，每个元素是对应行的最大值
                    new_probs = new_probs.unsqueeze(1)  # 形状变为[5,1]

                    # 检查是否有攻击成功的样本
                    for source_id, uid, pred, prob in zip(new_source_ids, new_uid, new_preds, new_probs):
                        if pred != label:
                            try:
                                # 将tensor转换为普通Python列表
                                ids_list = source_id.cpu().tolist()
                                
                                # 创建要保存的JSON对象
                                sample = {
                                    "project": "attack_train",
                                    "commit_id": "00000",
                                    "target": label.item(),  # 使用原始标签
                                    "func": ids_list,  # 直接保存token ids列表
                                    "idx": self.attack_sample_counter
                                }
                                
                                # 将样本追加到文件
                                with open(self.args.attack_train, 'a') as f:
                                    f.write(json.dumps(sample) + '\n')
                                    
                                # 递增计数器
                                self.attack_sample_counter += 1
                                
                            except Exception as e:
                                print(f"保存对抗样本错误: {e}")
                            
                            return True, pred, 1

                    # 如果没有成功的攻击，选择最佳候选项（概率最小的）
                    best_idx = torch.argmin(new_probs)
                    if new_probs[best_idx] < old_prob:
                        source_ids = new_source_ids[best_idx].unsqueeze(0)
                        uids[new_uid[best_idx]] = uids.pop(k)
                        n_stop = 0
                        made_changes = True
                        old_prob = new_probs[best_idx]
                    else:
                        n_stop += 1
            
            # 如果本轮中没有做任何有效替换，则退出循环
            if not made_changes:
                break
            
        return False, label, 2

    def attack_all(self, args, train_dataset, n_candidate=100, n_iter=20):
        """
        对训练数据集进行攻击，生成对抗样本用于训练
        """
        # 检查必要参数
        if not hasattr(args, 'attack_train') or not args.attack_train:
            print("错误: 未指定args.attack_train路径。无法保存对抗样本。")
            return {}
            
        attack_file_path = args.attack_train
        
        # 在开始攻击前清空文件
        print(f"清空或创建对抗样本文件: {attack_file_path}")
        with open(attack_file_path, 'w') as f:
            pass  # 以写入模式打开并立即关闭，会清空文件内容
        
        # 重置计数器为固定值
        self.attack_sample_counter = 30000
        print(f"attack_sample_counter已重置为: {self.attack_sample_counter}")
        
        # 设置对抗样本生成上限
        max_adversarial_samples = getattr(args, 'max_adversarial_samples', 6000)
        current_adversarial_count = 0
        
        # 连续失败计数器
        consecutive_failures = 0
        max_consecutive_failures = getattr(args, 'max_consecutive_failures', 200)
        
        n_succ = 0
        total_time = 0
        
        # 创建一个索引列表并随机打乱
        indices = list(range(len(train_dataset)))
        random.shuffle(indices)
        
        # 按照随机索引遍历数据集
        for idx in tqdm(indices, total=len(indices), desc="生成对抗样本"):
            # 检查是否达到对抗样本上限
            if current_adversarial_count >= max_adversarial_samples:
                print(f"已生成{max_adversarial_samples}个对抗样本，停止生成")
                break
            # 检查是否连续失败次数过多
            if consecutive_failures >= max_consecutive_failures:
                print(f"连续{max_consecutive_failures}次生成失败，停止生成")
                break
            
            # 直接获取对应索引的样本
            sample = train_dataset[idx]
            inputs = sample[0].unsqueeze(0).to(args.device)  # 添加批次维度
            label = sample[1].unsqueeze(0).to(args.device)   # 添加批次维度
            
            # 获取对应的uid并确保它是一个有效的字典
            current_uid = train_dataset.uids[idx]
            
            # 检查current_uid是否为有效数据类型
            if not isinstance(current_uid, dict) and current_uid != 0:
                # consecutive_failures += 1
                continue
            
            # 如果uid为0（可能是特殊值），创建一个空字典，这样攻击函数会跳过它
            if current_uid == 0:
                current_uid = {}
                
            start_time = time.time()
            success, pred, flag = self.attack(inputs, label, current_uid, n_candidate, n_iter)
            
            if success and flag == 1:  # 攻击成功且生成了新样本
                current_adversarial_count += 1
                n_succ += 1
                total_time += time.time() - start_time
                consecutive_failures = 0  # 重置连续失败计数器
            elif flag == 2:  # 攻击失败
                consecutive_failures += 1
            # elif flag == 0:  # 原始预测错误
            #     consecutive_failures += 1
        
        print(f"完成生成对抗样本。总共生成: {current_adversarial_count}")
        
        return {
            "success_rate": n_succ/len(train_dataset) if len(train_dataset) > 0 else 0,
            "avg_time": total_time/n_succ if n_succ > 0 else float('nan'),
            "generated_adversarial_samples": current_adversarial_count
        }

class InsAttackertrain(object):
    
    def __init__(self, model, my_tokenizer, args):
        self.model = model
        self.my_tokenizer = my_tokenizer
        self.args = args
        self.insM = InsModifier(classifier=model,
                               my_tokenizer=my_tokenizer,
                               poses=None) # 等待攻击时初始化
        
        # 初始化攻击样本计数器
        self.attack_sample_counter = 30000
        
        # 确保attack_train文件所在目录存在
        if hasattr(args, 'attack_train') and args.attack_train:
            attack_train_dir = os.path.dirname(args.attack_train)
            if attack_train_dir and not os.path.exists(attack_train_dir):
                os.makedirs(attack_train_dir, exist_ok=True)
    
    def Shape_processing(self, code_tokens):
        code_tokens = code_tokens[: self.args.block_size - 2]
        source_ids = [0] + code_tokens + [2]
        padding_length = self.args.block_size - len(source_ids)
        source_ids += [1] * padding_length
        return source_ids         

    def attack(self, code_tokens, label, poses, n_candidate=5, n_iter=20):
        self.insM.initInsertDict(poses)
        iter = 0  # 初始化迭代次数
        n_stop = 0  # 初始化停止计数器
        
        # 检查poses是否为空
        if len(poses) == 0:
            return False, label, 0        
            
        # 原始数据处理
        device = label.device
        source_ids = self.Shape_processing(code_tokens)
        source_ids = torch.tensor(source_ids).unsqueeze(0).to(device)
        
        # 获取原始预测
        with torch.no_grad():
            logit = self.model(source_ids)
            def getClass(x):
                return x.index(max(x))
            
            # 将预测转换为pandas Series并应用getClass函数
            probs = pd.Series(logit.tolist())
            preds = probs.apply(getClass)
            # 重置索引
            preds.reset_index(drop=True, inplace=True)
            
            # 将 preds 转换为 numpy 数组
            preds_numpy = preds.to_numpy()

            # 获取 label 所在的设备
            device = label.device

            # 将 numpy 数组转换为相同设备上的 tensor
            preds_tensor = torch.tensor(preds_numpy, device=device)
            old_pred = preds_tensor
            old_prob = logit
            old_prob, _ = torch.max(old_prob, dim=1)
            
        if old_pred != label:
            return True, old_pred, 0  # flag=0 表示原始预测错误

        while iter < n_iter:
            iter += 1
            n_candidate_ins = n_candidate          
            
            # 生成插入后的新输入数据
            new_code_tokens_add, new_insertDict_add = self.insM.insert(code_tokens, n_candidate_ins)
            new_code_tokens = new_code_tokens_add
            new_insertDict = new_insertDict_add

            if new_code_tokens == []:  # 如果没有有效的候选数据
                n_stop += 1  # 增加停止计数器
                continue  # 跳过本次迭代
                
            # 处理新代码tokens为模型输入
            new_source_ids = []
            for code_tokens in new_code_tokens:
                source_ids = self.Shape_processing(code_tokens)
                new_source_ids.append(source_ids)
            
            # 将列表转换为tensor
            new_source_ids = torch.tensor(new_source_ids).to(device)
            
            with torch.no_grad():
                outputs = self.model(new_source_ids)  # [batch_size]
                def getClass(x):
                    return x.index(max(x))
                
                # 将预测转换为pandas Series并应用getClass函数
                probs = pd.Series(outputs.tolist())
                preds = probs.apply(getClass)
                # 重置索引
                preds.reset_index(drop=True, inplace=True)
                
                # 将 preds 转换为 numpy 数组
                preds_numpy = preds.to_numpy()

                # 获取 label 所在的设备
                device = label.device

                # 将 numpy 数组转换为相同设备上的 tensor
                preds_tensor = torch.tensor(preds_numpy, device=device)
                new_preds = preds_tensor

                # 获取outputs中每行的最大值，保持维度
                new_probs, _ = torch.max(outputs, dim=1)  # 形状变为[5]，每个元素是对应行的最大值
                new_probs = new_probs.unsqueeze(1)  # 形状变为[5,1]

                # 检查是否有攻击成功的样本
                for source_id, pred in zip(new_source_ids, new_preds):
                    if pred != label:
                        try:
                            # 将tensor转换为普通Python列表
                            ids_list = source_id.cpu().tolist()

                            # 创建要保存的JSON对象
                            sample = {
                                "project": "attack_train_insertion",
                                "commit_id": "00000_ins",
                                "target": label.item(),  # 使用原始标签
                                "func": ids_list,  # 直接保存token ids列表
                                "idx": self.attack_sample_counter
                            }
                            
                            # 将样本追加到文件
                            with open(self.args.attack_train, 'a') as f:
                                f.write(json.dumps(sample) + '\n')
                                
                            # 递增计数器
                            self.attack_sample_counter += 1
                            
                        except Exception as e:
                            print(f"保存对抗样本错误: {e}")
                        
                        return True, pred, 1  # flag=1 表示攻击成功并保存

                # 如果没有成功的攻击，选择最佳候选项（概率最小的）
                best_idx = torch.argmin(new_probs)
                if new_probs[best_idx] < old_prob:
                    self.insM.insertDict = new_insertDict[best_idx]
                    n_stop = 0
                    old_prob = new_probs[best_idx]
                else:
                    n_stop += 1
                    
                if n_stop >= len(new_source_ids):
                    iter = n_iter
                    break
        return False, label, 2  # flag=2 表示攻击失败

    def attack_all(self, args, train_dataset, n_candidate=100, n_iter=20):
        """
        对训练数据集进行攻击，生成对抗样本用于训练
        """
        # 检查必要参数
        if not hasattr(args, 'attack_train') or not args.attack_train:
            print("错误: 未指定args.attack_train路径。无法保存对抗样本。")
            return {}
            
        attack_file_path = args.attack_train
        
        # 在开始攻击前清空文件
        print(f"清空或创建对抗样本文件: {attack_file_path}")
        with open(attack_file_path, 'w') as f:
            pass  # 以写入模式打开并立即关闭，会清空文件内容
        
        # 重置计数器为固定值
        self.attack_sample_counter = 30000
        print(f"attack_sample_counter已重置为: {self.attack_sample_counter}")
        
        # 设置对抗样本生成上限
        max_adversarial_samples = getattr(args, 'max_adversarial_samples', 6000)
        current_adversarial_count = 0
        
        # 连续失败计数器
        consecutive_failures = 0
        max_consecutive_failures = getattr(args, 'max_consecutive_failures', 200)
        
        n_succ = 0
        total_time = 0
        
        # 创建一个索引列表并随机打乱
        indices = list(range(len(train_dataset)))
        random.shuffle(indices)
        
        # 按照随机索引遍历数据集
        for idx in tqdm(indices, total=len(indices), desc="生成插入对抗样本"):
            # 检查是否达到对抗样本上限
            if current_adversarial_count >= max_adversarial_samples:
                print(f"已生成{max_adversarial_samples}个对抗样本，停止生成")
                break
            # 检查是否连续失败次数过多
            if consecutive_failures >= max_consecutive_failures:
                print(f"连续{max_consecutive_failures}次生成失败，停止生成")
                break
            
            # 直接获取对应索引的样本
            sample = train_dataset[idx]
            label = sample[1].unsqueeze(0).to(args.device)   # 添加批次维度
            current_code_tokens = train_dataset.code_tokens[idx]
            inputs = current_code_tokens
            # inputs = self.my_tokenizer.encode(current_code_tokens).ids
            # 获取对应的instab并确保它是一个有效的字典
            current_poses = train_dataset.instab[idx]
            
            # 如果current_poses为0（可能是特殊值），创建一个空字典，这样攻击函数会跳过它
            if current_poses == 0:
                current_poses = {}
            
            # 执行攻击
            start_time = time.time()
            success, pred, flag = self.attack(inputs, label, current_poses, n_candidate, n_iter)
            if success and flag == 1:  # 攻击成功且生成了新样本
                current_adversarial_count += 1
                n_succ += 1
                total_time += time.time() - start_time
                consecutive_failures = 0  # 重置连续失败计数器
            elif flag == 2:  # 攻击失败
                consecutive_failures += 1
            # elif flag == 0:  # 原始预测错误
            #     consecutive_failures += 1
        
        print(f"完成生成插入对抗样本。总共生成: {current_adversarial_count}")
        
        return {
            "success_rate": n_succ/len(train_dataset),
            "avg_time": total_time/n_succ if n_succ > 0 else float('nan'),
            "total_samples": current_adversarial_count
        }

class AttackerRandomtrain(object):
    
    def __init__(self, model, my_tokenizer, uids, args):
        self.model = model
        self.my_tokenizer = my_tokenizer
        self.tokenM = TokenModifier(
            classifier=model,
            loss=torch.nn.CrossEntropyLoss(),
            uids=uids,
            my_tokenizer=my_tokenizer,
            args = args
        )
        self.args = args
        
        # 初始化攻击样本计数器
        self.attack_sample_counter = 30000
        
        # 确保attack_train文件所在目录存在
        if hasattr(args, 'attack_train') and args.attack_train:
            attack_train_dir = os.path.dirname(args.attack_train)
            if attack_train_dir and not os.path.exists(attack_train_dir):
                os.makedirs(attack_train_dir, exist_ok=True)
    
    def attack(self, source_ids, label, uids, n_iter=20):
        """
        执行随机攻击
        Args:
            source_ids: 输入序列的token ids
            label: 真实标签
            uids: 可替换的标识符信息
            n_iter: 最大攻击迭代次数
        """
        iter = 0
        n_stop = 0
        
        # 检查uids是否为有效的字典
        if not isinstance(uids, dict) or len(uids) == 0:
            return False, label, 0
        
        # 获取原始预测
        with torch.no_grad():
            logit = self.model(source_ids)
            def getClass(x):
                return x.index(max(x))
            
            # 将预测转换为pandas Series并应用getClass函数
            probs = pd.Series(logit.tolist())
            preds = probs.apply(getClass)
            # 重置索引
            preds.reset_index(drop=True, inplace=True)
            
            # 将 preds 转换为 numpy 数组
            preds_numpy = preds.to_numpy()

            # 获取 label 所在的设备
            device = label.device

            # 将 numpy 数组转换为相同设备上的 tensor
            preds_tensor = torch.tensor(preds_numpy, device=device)
            old_pred = preds_tensor
            old_prob = logit
            old_prob, _ = torch.max(old_prob, dim=1)
            
        if old_pred != label:
            return True, old_pred, 0
            
        while iter < n_iter:
            # 获取所有可替换的标识符
            keys = list(uids.keys())
            made_changes = False  # 标记是否在本轮中做了有效替换
            
            for k in keys:
                if iter >= n_iter:
                    break
                if n_stop >= len(uids):
                    iter = n_iter
                    break
                    
                iter += 1
                # 随机替换标识符
                new_source_ids, new_uid = self.tokenM.rename_uid_random(source_ids, k, keys)
                
                if new_source_ids is None:
                    n_stop += 1
                    continue
                    
                # 获取新预测
                with torch.no_grad():
                    outputs = self.model(new_source_ids)
                    def getClass(x):
                        return x.index(max(x))
                    
                    # 将预测转换为pandas Series并应用getClass函数
                    probs = pd.Series(outputs.tolist())
                    preds = probs.apply(getClass)
                    # 重置索引
                    preds.reset_index(drop=True, inplace=True)
                    
                    # 将 preds 转换为 numpy 数组
                    preds_numpy = preds.to_numpy()

                    # 获取 label 所在的设备
                    device = label.device

                    # 将 numpy 数组转换为相同设备上的 tensor
                    preds_tensor = torch.tensor(preds_numpy, device=device)
                    new_preds = preds_tensor

                    # 获取outputs中每行的最大值，保持维度
                    new_probs, _ = torch.max(outputs, dim=1)  # 形状变为[1]，每个元素是对应行的最大值
                    new_probs = new_probs.unsqueeze(1)  # 形状变为[1,1]
                
                # 检查是否攻击成功
                if new_preds != label:
                    try:
                        # 将tensor转换为普通Python列表并移除批次维度
                        ids_list = new_source_ids.cpu().squeeze(0).tolist()
                        
                        # 创建要保存的JSON对象
                        sample = {
                            "project": "attack_train_random",
                            "commit_id": "00000_random",
                            "target": label.item(),  # 使用原始标签
                            "func": ids_list,  # 直接保存token ids列表，已经移除批次维度
                            "idx": self.attack_sample_counter
                        }
                        
                        # 将样本追加到文件
                        with open(self.args.attack_train, 'a') as f:
                            f.write(json.dumps(sample) + '\n')
                            
                        # 递增计数器
                        self.attack_sample_counter += 1
                        
                    except Exception as e:
                        print(f"保存对抗样本错误: {e}")
                    
                    return True, new_preds, 1
                    
                # 更新当前最佳结果
                if new_probs < old_prob:
                    source_ids = new_source_ids
                    uids[new_uid] = uids.pop(k)
                    n_stop = 0
                    made_changes = True  # 标记本轮做了有效替换
                    # print(f"acc\t{k} => {new_uid}\t\t{label.item()}({old_prob.item():.5f}) => {label.item()}({new_probs.item():.5f})")
                    old_prob = new_probs
                else:
                    n_stop += 1
            
            # 如果本轮中没有做任何有效替换，则退出循环
            if not made_changes:
                break
                    
        return False, label, 2

    def attack_all(self, args, train_dataset, n_iter=20):
        """
        对训练数据集进行攻击，生成对抗样本用于训练
        """
        # 检查必要参数
        if not hasattr(args, 'attack_train') or not args.attack_train:
            print("错误: 未指定args.attack_train路径。无法保存对抗样本。")
            return {}
            
        attack_file_path = args.attack_train
        
        # 在开始攻击前清空文件
        print(f"清空或创建对抗样本文件: {attack_file_path}")
        with open(attack_file_path, 'w') as f:
            pass  # 以写入模式打开并立即关闭，会清空文件内容
        
        # 重置计数器为固定值
        self.attack_sample_counter = 30000
        print(f"attack_sample_counter已重置为: {self.attack_sample_counter}")
        
        # 设置对抗样本生成上限
        max_adversarial_samples = getattr(args, 'max_adversarial_samples', 6000)
        current_adversarial_count = 0
        
        # 连续失败计数器
        consecutive_failures = 0
        max_consecutive_failures = getattr(args, 'max_consecutive_failures', 200)
        
        n_succ = 0
        total_time = 0
        
        # 创建一个索引列表并随机打乱
        indices = list(range(len(train_dataset)))
        random.shuffle(indices)
        
        # 按照随机索引遍历数据集
        for idx in tqdm(indices, total=len(indices), desc="生成随机对抗样本"):
            # 检查是否达到对抗样本上限
            if current_adversarial_count >= max_adversarial_samples:
                print(f"已生成{max_adversarial_samples}个对抗样本，停止生成")
                break
            # 检查是否连续失败次数过多
            if consecutive_failures >= max_consecutive_failures:
                print(f"连续{max_consecutive_failures}次生成失败，停止生成")
                break
            
            # 直接获取对应索引的样本
            sample = train_dataset[idx]
            inputs = sample[0].unsqueeze(0).to(args.device)  # 添加批次维度
            label = sample[1].unsqueeze(0).to(args.device)   # 添加批次维度
            
            # 获取对应的uid并确保它是一个有效的字典
            current_uid = train_dataset.uids[idx]
            
            # 检查current_uid是否为有效数据类型
            if not isinstance(current_uid, dict) and current_uid != 0:
                # consecutive_failures += 1
                continue
            
            # 如果uid为0（可能是特殊值），创建一个空字典，这样攻击函数会跳过它
            if current_uid == 0:
                current_uid = {}
                
            start_time = time.time()
            success, pred, flag = self.attack(inputs, label, current_uid, n_iter)
            
            if success and flag == 1:  # 攻击成功且生成了新样本
                current_adversarial_count += 1
                n_succ += 1
                total_time += time.time() - start_time
                consecutive_failures = 0  # 重置连续失败计数器
            elif flag == 2:  # 攻击失败
                consecutive_failures += 1
            # elif flag == 0:  # 原始预测错误
            #     consecutive_failures += 1
        
        print(f"完成生成随机对抗样本。总共生成: {current_adversarial_count}")
        
        return {
            "success_rate": n_succ/len(train_dataset) if len(train_dataset) > 0 else 0,
            "avg_time": total_time/n_succ if n_succ > 0 else float('nan'),
            "generated_adversarial_samples": current_adversarial_count
        }