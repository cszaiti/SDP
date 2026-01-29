# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
from torch.nn import CrossEntropyLoss, MSELoss


class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        self.embed = self.encoder.roberta.embeddings.word_embeddings

    def forward(self, input_ids=None, labels=None, weight_V=None):
        qwer = self.encoder(input_ids, attention_mask=input_ids.ne(1))[0]
        # hidden_states = qwer.hidden_states
        # last_hidden_state = hidden_states[-1]
        # logits = self.encoder.classifier(last_hidden_state)
        # logits = outputs
        prob = torch.sigmoid(qwer)
        if prob.shape[1] != 1:
            prob = torch.mean(prob, axis=[1, 2]).unsqueeze(1)
#        print(prob.shape)
        if labels is not None:
            labels = labels.float()
            # sample_weights = 0.1 * weight_V.float() + 1.0
            loss = torch.log(prob[:, 0] + 1e-10) * labels + torch.log((1 - prob)[:, 0] + 1e-10) * (1 - labels)
            # loss = loss * sample_weights 
            loss = -loss.mean()
            return loss, prob
        else:
            return prob
    def grad(self, inputs, labels):
        # 清除所有梯度
        self.encoder.zero_grad()
        
        # 确保嵌入层梯度会被保留
        self.embed.weight.retain_grad()
        
        # 前向传播计算损失
        loss, _ = self.forward(inputs, labels)
        
        # 反向传播计算梯度
        loss.backward()
        
        # 直接返回嵌入层权重梯度
        return self.embed.weight.grad