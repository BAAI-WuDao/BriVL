# BriVL


BriVL (Bridging Vision and Language Model) 是首个中文通用图文多模态大规模预训练模型。BriVL模型在图文检索任务上有着优异的效果，超过了同期其他常见的多模态预训练模型（例如UNITER、CLIP）。

BriVL论文：[WenLan: Bridging Vision and Language by Large-Scale Multi-Modal Pre-Training](https://arxiv.org/abs/2103.06561)


# 适用场景

适用场景示例：图像检索文本、文本检索图像、图像标注、图像零样本分类、作为其他下游多模态任务的输入特征等。

# 技术特色

1. BriVL使用对比学习算法将图像和文本映射到了同一特征空间，可用于弥补图像特征和文本特征之间存在的隔阂。
2. 基于视觉-语言弱相关的假设，除了能理解对图像的描述性文本外，也可以捕捉图像和文本之间存在的抽象联系。
3. 图像编码器和文本编码器可分别独立运行，有利于实际生产环境中的部署。 

# 下载专区


源码链接 [立即前往](https://)

| 模型      | 语言 | 参数量（单位：亿） | 文件（file）                | 
| --------- | ---- | ------------------ | --------------------------- |
| BriVL-1.0  | 中文 | 10亿                 | BriVL-1.0-5500w.tar| 



# 使用BriVL

### 搭建环境

```
# 环境要求
lmdb==0.99
timm==0.4.12
easydict==1.9
pandas==1.2.4
jsonlines==2.0.0
tqdm==4.60.0
torchvision==0.9.1
numpy==1.20.2
torch==1.8.1
transformers==4.5.1
msgpack_numpy==0.4.7.1
msgpack_python==0.5.6
Pillow==8.3.1
PyYAML==5.4.1
```

配置要求在requirements.txt中，可使用下面的命令：


```
pip install -r requirements.txt
```


### 特征提取与计算检索结果

```
cd evaluation/
bash test_xyb.sh
```

### 数据解释
现已放入3个图文对示例:

```
./data/imgs  # 放入图像
./data/jsonls # 放入图文对描述
```

# 引用BriVL

```
@article{DBLP:journals/corr/abs-2103-06561,
  author    = {Yuqi Huo and
               Manli Zhang and
               Guangzhen Liu and
               Haoyu Lu and
               Yizhao Gao and
               Guoxing Yang and
               Jingyuan Wen and
               Heng Zhang and
               Baogui Xu and
               Weihao Zheng and
               Zongzheng Xi and
               Yueqian Yang and
               Anwen Hu and
               Jinming Zhao and
               Ruichen Li and
               Yida Zhao and
               Liang Zhang and
               Yuqing Song and
               Xin Hong and
               Wanqing Cui and
               Dan Yang Hou and
               Yingyan Li and
               Junyi Li and
               Peiyu Liu and
               Zheng Gong and
               Chuhao Jin and
               Yuchong Sun and
               Shizhe Chen and
               Zhiwu Lu and
               Zhicheng Dou and
               Qin Jin and
               Yanyan Lan and
               Wayne Xin Zhao and
               Ruihua Song and
               Ji{-}Rong Wen},
  title     = {WenLan: Bridging Vision and Language by Large-Scale Multi-Modal Pre-Training},
  journal   = {CoRR},
  volume    = {abs/2103.06561},
  year      = {2021},
  url       = {https://arxiv.org/abs/2103.06561},
  archivePrefix = {arXiv},
  eprint    = {2103.06561},
  timestamp = {Tue, 03 Aug 2021 12:35:30 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2103-06561.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```



