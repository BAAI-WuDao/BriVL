#!/bin/bash
load_checkpoint=../../BriVL-pretrain-model/BriVL-1.0-5500w.pth # load model path
# for retrieval 
feat_save_dir=../logs/saves/xyb # save feature dir
python3 XYB_box_extract.py --feat_save_dir ${feat_save_dir} --load_checkpoint ${load_checkpoint}
python3 cal_xyb_retrieval.py --feat_load_dir ${feat_save_dir}

