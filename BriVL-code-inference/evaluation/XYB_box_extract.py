import sys
import os
# sys.path = sys.path[:-2]
sys.path.append(os.path.abspath(os.path.dirname(os.path.realpath(__file__))+'/'+'..')) 
import os
import time
import argparse
import torch
import json
from tqdm import tqdm
import math
import numpy as np
import random

from utils import getLanMask
from models import build_network
from dataset import build_moco_dataset
from utils.config import cfg_from_yaml_file, cfg

parser = argparse.ArgumentParser()
parser.add_argument('--pretrainRes', type=str, default='../logs/pretrain/FixRes.pth')
parser.add_argument('--load_checkpoint', type=str, default=None)
parser.add_argument('--data_dir', type=str, default='/data1/clean_data/AIC')

parser.add_argument('--feat_save_dir', type=str, default=None)
parser.add_argument('--gpu_ids', type=str, default='7')
parser.add_argument('--option', type=str, default='img_text')
parser.add_argument('--Nsamples', type=int, default=None)
parser.add_argument('--seed', type=int, default=222)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--cfg_file', type=str, default='../cfg/test_xyb.yml')
args = parser.parse_args()

cfg_from_yaml_file(args.cfg_file, cfg)

# image -> text retrieval , 0 for <image2text, text>, 1 for <image, text2image>
# also extract seq_len

param_group = { 'img_text': {'img_fname':'np_img.npy', 'text_fname':'np_text.npy'}}

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
torch.manual_seed(args.seed) # cpu
torch.cuda.manual_seed(args.seed) #gpu
np.random.seed(args.seed) #numpy
random.seed(args.seed) #random and transforms
torch.backends.cudnn.deterministic=True # cudnn
torch.cuda.set_device(args.gpu)

dataloader_test = build_moco_dataset(args, cfg, is_training=False)

model = build_network(cfg.MODEL)
model = model.cuda()
model_component = torch.load(args.load_checkpoint, map_location=torch.device('cpu'))
model.learnable.load_state_dict(model_component['learnable'])    ####### only save learnable
model = torch.nn.DataParallel(model)
model.eval()

if not os.path.exists(args.feat_save_dir):
    os.makedirs(args.feat_save_dir)
    print('Successfully create feature save dir {} !'.format(args.feat_save_dir))

print('Load model from {:s}'.format(args.load_checkpoint))
print('Save features to dir {:s}'.format(args.feat_save_dir))
with torch.no_grad():

    num_samples = len(dataloader_test)
    np_text, np_img = None, None
    for idx, batch in enumerate(tqdm(dataloader_test)):

        # data 
        imgs = batch[0]  # <batchsize, 3, image_size, image_size>
        #print(imgs.size())
        img_lens = batch[1].view(-1)
        texts = batch[2]  # <batchsize, 5, max_textLen>
        text_lens = batch[3] # <batchsize, 5, >
        image_boxs = batch[4] # <BSZ, 36, 4>

        bsz, textlen = texts.size(0), texts.size(1)
        # get image mask
        # print(imgs.size(), texts.size())
        imgMask = getLanMask(img_lens, cfg.MODEL.MAX_IMG_LEN)
        imgMask = imgMask.cuda()

        # get language mask
        textMask = getLanMask(text_lens, cfg.MODEL.MAX_TEXT_LEN)
        textMask = textMask.cuda()

        imgs = imgs.cuda()
        texts = texts.cuda()
        image_boxs = image_boxs.cuda() # <BSZ, 36, 4>


        text_lens = text_lens.cuda() ############
        feature_group = model(imgs, texts, imgMask, textMask, text_lens, image_boxs, is_training=False)
        img, text = feature_group[args.option] # img2text_text / img_text2img 

        if np_img is None:
            np_img = img.cpu().numpy() # <bsz, featdim>
            np_text = text.cpu().numpy() # <bsz, cap_num, featdim>
    
        else:
            np_img = np.concatenate((np_img, img.cpu().numpy()), axis=0)
            np_text = np.concatenate((np_text, text.cpu().numpy()), axis=0)

    fn_img = os.path.join(args.feat_save_dir, param_group[args.option]['img_fname']) # 'np_img2text.npy' / 'np_img.npy'
    fn_text = os.path.join(args.feat_save_dir, param_group[args.option]['text_fname']) # 'np_text.npy' / 'np_text2img.npy'

    np.save(fn_img, np_img) 
    np.save(fn_text, np_text) 


        
