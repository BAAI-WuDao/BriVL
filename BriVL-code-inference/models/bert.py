import torch
import torch.nn as nn
from transformers import AutoModel


class Bert(nn.Module):

    def __init__(self, args):
        super(Bert, self).__init__()
        self.args = args
        #self.bert = AutoModel.from_pretrained('hfl/chinese-bert-wwm-ext')
        self.bert =  AutoModel.from_pretrained(args.ENCODER) 
        #self.bert = AutoModel.from_pretrained('bert-base-chinese')

    def forward(self, x):
        # y = torch.ones((int(self.args.batch_size/4), self.args.max_textLen, self.args.textFea_dim),device=x.device)   
        y = self.bert(x, return_dict=True).last_hidden_state
        return y
