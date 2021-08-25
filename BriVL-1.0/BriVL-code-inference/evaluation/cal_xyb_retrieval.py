import os
import numpy as np
import random
from tqdm import tqdm
import argparse
import torch
import sys 
sys.path.append(os.path.abspath(os.path.dirname(os.path.realpath(__file__))+'/'+'..')) 


parser = argparse.ArgumentParser()
parser.add_argument('--feat_load_dir', type=str, default='./logs/feature/ance_trip')
parser.add_argument('--seed', type=int, default=222)
parser.add_argument('--gpu', type=str, default='3')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

random.seed(args.seed)
np.random.seed(args.seed)

rootDir = args.feat_load_dir
np_img = np.load(os.path.join(rootDir, 'np_img.npy')).astype(np.float64)# <N, featdim>
feat_dim = np_img.shape[-1]
np_text = np.load(os.path.join(rootDir, 'np_text.npy')).astype(np.float64) # <N, featdim>
N = np_text.shape[0]

np_text_mean = np_text
np_img_mean = np_img
img = torch.from_numpy(np_img_mean).cuda() # <N, featdim>
text = torch.from_numpy(np_text_mean).cuda() # <N, featdim>

print(img.size(), text.size(), N)
scores = torch.zeros((N, N), dtype=torch.float32).cuda()
print('Pair-to-pair: calculating scores')
for i in tqdm(range(N)): # row: image  col: text
    scores[i, :] = torch.sum(img[i] * text, -1)

# ground truth 
recall_k_s = [1] # [1, 5, 10]
GT_label = torch.arange(0, N).view(N, 1).cuda()

# img2text
print('Option is img2text')
logits = scores
indices = torch.argsort(logits, descending=True)
gt_rank = (indices == GT_label).float() 
gt_rank = gt_rank.cumsum(dim=1)
gt_rank_i2t = gt_rank.cpu().numpy()

for recall_k in recall_k_s:
    recall = 100 * (np.sum(gt_rank_i2t[:, recall_k - 1]) / N)
    print('Recall@{:d}:  {:.2f}%'.format(recall_k, recall))

# text2img
print('Option is text2img')
logits = scores.T
indices = torch.argsort(logits, descending=True) 
gt_rank = (indices == GT_label).float() 
gt_rank = gt_rank.cumsum(dim=1)
gt_rank_t2i = gt_rank.cpu().numpy()

for recall_k in recall_k_s:
    recall = 100 * (np.sum(gt_rank_t2i[:, recall_k - 1]) / N)
    print('Recall@{:d}:  {:.2f}%'.format(recall_k, recall))
    

