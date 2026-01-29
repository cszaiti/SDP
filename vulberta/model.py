# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F

class myCNN(nn.Module):
    def __init__(self, model, EMBED_SIZE, EMBED_DIM):
        super(myCNN,self).__init__()
        
        # pretrained_weights = RobertaModel.from_pretrained('/root/workspace/zlt/vulberta/models/VulBERTa/').embeddings.word_embeddings.weight

        self.embed = nn.Embedding.from_pretrained(model,
                                                  freeze=True,
                                                  padding_idx=1)
        
        self.conv1 = nn.Conv1d(in_channels=EMBED_DIM, out_channels=200, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=EMBED_DIM, out_channels=200, kernel_size=4)
        self.conv3 = nn.Conv1d(in_channels=EMBED_DIM, out_channels=200, kernel_size=5)

        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(200*3,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,2)
    #     cw = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced',classes=[0,1],y=m1.target.tolist())
    # # 将计算出的权重转换为PyTorch张量，用于损失函数
    #     c_weights = torch.FloatTensor([cw[0], cw[1]])
        c_weights = torch.FloatTensor([0.923, 1.09])
        self.criterion = nn.CrossEntropyLoss(weight=c_weights)
    
    def forward(self, x,labels=None):
        x = self.embed(x)
        x = x.permute(0,2,1)

        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        x3 = F.relu(self.conv3(x))
        
        x1 = F.max_pool1d(x1, x1.shape[2])
        x2 = F.max_pool1d(x2, x2.shape[2])
        x3 = F.max_pool1d(x3, x3.shape[2])
        
        x = torch.cat([x1,x2,x3],dim=1)
        x = x.flatten(1)
        x = self.dropout(x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        prob = self.fc3(x)
        


        # prob = probs = pd.Series(logits)
        # if prob.shape[1] != 1:
        #     prob = torch.mean(prob, axis=[1, 2]).unsqueeze(1)
#        print(prob.shape)
        if labels is not None:
            # labels = labels.tolist()
            # sample_weights = 0.1 * weight_V.float() + 1.0
            loss = self.criterion(prob, labels)
            # loss = loss * sample_weights 
            # loss = loss.mean()
            return loss, prob
        else:
            return prob
    def grad(self, inputs, labels):
        # 清除所有梯度
        self.zero_grad()
        
        # 临时解除嵌入层的冻结状态以计算梯度
        original_embedding_state = self.embed.weight.requires_grad
        self.embed.weight.requires_grad = True
        
        # 确保嵌入层梯度会被保留
        self.embed.weight.retain_grad()
        
        # 前向传播计算损失
        loss, _ = self.forward(inputs, labels)
        
        # 反向传播计算梯度
        loss.backward()
        
        # 获取嵌入层的梯度
        embedding_grad = self.embed.weight.grad.clone()
        
        # 恢复嵌入层的原始冻结状态
        self.embed.weight.requires_grad = original_embedding_state
        
        # 返回嵌入层权重梯度
        return embedding_grad