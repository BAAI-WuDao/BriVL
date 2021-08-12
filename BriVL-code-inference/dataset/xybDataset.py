import os
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import json
from transformers import AutoTokenizer
import random
from PIL import ImageFilter
import msgpack
import msgpack_numpy as m

m.patch()
from io import BytesIO
import lmdb
import jsonlines
import pandas
from transformers import BertTokenizer

# tf_efficientnet_b7_ns
def visual_transforms_box(is_train=True, new_size = 456):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if is_train:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((new_size, new_size)),   ##########
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            normalize])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((new_size, new_size)),
            normalize])

class XYBDataset_all(data.Dataset): 
    def __init__(self, cfg, args, phase=None):

        self.cfg = cfg
        self.args = args
        self.text_transform = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
        
        # self.data_dir = cfg.DATASET.DATADIR
        # json_path = os.path.join(self.data_dir, cfg.DATASET.JSONPATH)
        json_path = cfg.DATASET.JSONPATH
        self.visual_transform = visual_transforms_box(True)
        self.imgnames = []
        self.bboxs = []
        self.sentences = []
        with open(json_path, "r", encoding="utf8") as f:
            for item in jsonlines.Reader(f): # [{'image_id':, 'captions':[,,], 'bbox':[[], []...]}, ....]
                self.imgnames.append(item['image_id'])
                self.bboxs.append(item['bbox'][: self.cfg.MODEL.MAX_IMG_LEN - 1])
                self.sentences.append(item['sentences'])
        print('Dataset length {}'.format(len(self.imgnames)))
        if phase == 'train':
            self.visual_transform = visual_transforms_box(True, cfg.MODEL.IMG_SIZE)
        else:
            print('Validation dataset')
            self.visual_transform = visual_transforms_box(False, cfg.MODEL.IMG_SIZE)


    def __len__(self):
        return len(self.imgnames)
    
    def __getitem__(self, index):
        new_size = self.cfg.MODEL.IMG_SIZE
        ################################## image 
        # img_path = os.path.join(self.data_dir , self.imgnames[index]) 
        img_path = self.imgnames[index]
        image = Image.open(img_path).convert('RGB')
        width, height = image.size

        img_box_s = []

        box_grid = self.cfg.MODEL.BOX_GRID
        for box_i in self.bboxs[index]: # bbox number:  self.cfg.MODEL.MAX_IMG_LEN-1
            x1, y1, x2, y2 = box_i[0] * (new_size/width), box_i[1] * (new_size/height), box_i[2] * (new_size/width), box_i[3] * (new_size/height)
            img_box_s.append(torch.from_numpy(np.array([x1, y1, x2, y2]).astype(np.float32)))   
        img_box_s.append(torch.from_numpy(np.array([0, 0, new_size, new_size]).astype(np.float32))) # bbox number:  self.cfg.MODEL.MAX_IMG_LEN
        
        valid_len = len(img_box_s)
        img_len = torch.full((1,), valid_len, dtype=torch.long)

        if valid_len < self.cfg.MODEL.MAX_IMG_LEN:
            for i in range(self.cfg.MODEL.MAX_IMG_LEN - valid_len):
                img_box_s.append(torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32)))   

        image_boxs = torch.stack(img_box_s, 0) # <36, box_grid>

        image = self.visual_transform(image)

        ################################## text 
        imgdir_prefix  = self.imgnames[index]
        sentence = self.sentences[index]
        text_dict = {}
        for s in sentence:
            text_dict[s[0]] = s[1]
        text = text_dict['sur_text']
        
        text_info = self.text_transform(text, padding='max_length', truncation=True,
                                        max_length=self.cfg.MODEL.MAX_TEXT_LEN, return_tensors='pt')
        text = text_info.input_ids.reshape(-1)
        text_len = torch.sum(text_info.attention_mask)
        del text_dict

        return image, img_len, text, text_len, image_boxs