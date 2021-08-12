import torch
import torch.nn as nn
from .fakeTransformer import FakeTransformer
from .bert import Bert
from utils import pairLoss, alignmentLoss, attAlignmentLoss, AlignTripLoss, SimpTripLoss, NCELoss
import torch.nn.functional as F
import timm
import numpy as np
import sys

class ImgLearnableEncoder(nn.Module):
    def __init__(self, model_cfg):

        super(ImgLearnableEncoder, self).__init__()

        self.backbone = timm.create_model(model_cfg.CNN, pretrained=True)
        self.model_cfg = model_cfg

        self.learnable = nn.ModuleDict()
        self.learnable['imgFC'] = FakeTransformer(model_cfg.IMG_FEATURE_DIM, model_cfg.IMG_FEATURE_DIM, model_cfg.IMG_FEATURE_DIM)
        img_encoder_layer = nn.TransformerEncoderLayer(d_model=model_cfg.IMG_FEATURE_DIM, nhead=model_cfg.IMG_TRANSFORMER_HEAD)
        self.learnable['imgAtt'] = nn.TransformerEncoder(img_encoder_layer, num_layers=model_cfg.IMG_TRANSFORMER_LAYER)

        self.learnable['max_pool'] = nn.Sequential(
                                                    nn.Conv2d(model_cfg.IMG_FEATURE_DIM, model_cfg.IMG_FEATURE_DIM, kernel_size=1),
                                                    nn.AvgPool2d(model_cfg.GRID_SIZE, stride=1)
                                                ) 

        self.init_param()

    def init_param(self):

        for name, param in self.backbone.named_parameters():
            # print('@@@@@@@@@@@@@@@@@@@@@@@')

            condition = 'blocks.6' not in name and 'blocks.5' not in name and 'blocks.4' not in name and 'blocks.3' not in name
            
            if condition:
                param.requires_grad = False
            else:
                print(name + ' need grads')
                param.requires_grad = True
        sys.stdout.flush()
            
        

    def roi_grid_pool(self, spatial_features_2d, rois):
        """
        Args:
            rois: (B, num_rois, 4)
            spatial_features_2d: (B, C, H, W)
        Returns:
            pooled_features : (B, num_rois, C)

        """
        batch_size = spatial_features_2d.size(0)
        rois = rois.detach()
        height, width = spatial_features_2d.size(2), spatial_features_2d.size(3) # 特征图的长宽

        #print(spatial_features_2d.size())
        down_sample_ratio = self.model_cfg.IMG_SIZE / height

        pooled_features_list = []
        torch.backends.cudnn.enabled = False
        for b_id in range(batch_size):
            # todo 这里有一个坐标系的转换需要做
            # Map global boxes coordinates to feature map coordinates
            x1 = rois[b_id, :, 0] / down_sample_ratio
            y1 = rois[b_id, :, 1] / down_sample_ratio
            x2 = rois[b_id, :, 2] / down_sample_ratio
            y2 = rois[b_id, :, 3] / down_sample_ratio
            #print(x1, y1, x2, y2)

            angle = torch.zeros((1), device=spatial_features_2d.device)  ##########

            cosa = torch.cos(angle)
            sina = torch.sin(angle)

            theta = torch.stack((
                (x2 - x1) / (width - 1) * cosa, (x2 - x1) / (width - 1) * (-sina), (x1 + x2 - width + 1) / (width - 1),
                (y2 - y1) / (height - 1) * sina, (y2 - y1) / (height - 1) * cosa, (y1 + y2 - height + 1) / (height - 1)
            ), dim=1).view(-1, 2, 3).float()

            grid_size = self.model_cfg.GRID_SIZE
            grid = nn.functional.affine_grid(
                theta,
                torch.Size((rois.size(1), spatial_features_2d.size(1), grid_size, grid_size))
            )

            pooled_features = nn.functional.grid_sample(
                spatial_features_2d[b_id].unsqueeze(0).expand(rois.size(1), spatial_features_2d.size(1), height, width),
                grid
            )
            pooled_features = self.learnable['max_pool'](pooled_features)
            pooled_features_list.append(pooled_features.squeeze())

        torch.backends.cudnn.enabled = True
        pooled_features = torch.stack(pooled_features_list, dim=0)

        return pooled_features

    def forward(self, imgFea, maskImages, image_boxs):

        feature_map = self.backbone.forward_features(imgFea)
        imgFea = self.roi_grid_pool(feature_map, image_boxs)

        imgFea = F.normalize(imgFea, p=2, dim=-1)
        imgFea = self.learnable['imgAtt'](imgFea.transpose(0, 1), src_key_padding_mask=(maskImages == 0)).transpose(0,1)

        tmpMask = torch.where(maskImages == 1, torch.tensor([1.0], device=maskImages.device),
                              torch.tensor([0.0], device=maskImages.device))
        imgFea = (imgFea * tmpMask.unsqueeze(-1)).sum(dim=1) / tmpMask.sum(dim=1).unsqueeze(-1)  # (bs, dim)

        imgFea = self.learnable['imgFC'](imgFea)
        return imgFea


class TextLearnableEncoder(nn.Module):
    def __init__(self, model_cfg):
        super(TextLearnableEncoder, self).__init__()

        self.backbone = Bert(model_cfg)
        self.model_cfg = model_cfg

        self.learnable = nn.ModuleDict()
        self.learnable['textFC'] = FakeTransformer(model_cfg.TEXT_FEATURE_DIM, model_cfg.IMG_FEATURE_DIM, model_cfg.IMG_FEATURE_DIM)
        text_encoder_layer = nn.TransformerEncoderLayer(d_model=model_cfg.TEXT_FEATURE_DIM, nhead=model_cfg.TEXT_TRANSFORMER_HEAD)
        self.learnable['textAtt'] = nn.TransformerEncoder(text_encoder_layer, num_layers=model_cfg.TEXT_TRANSFORMER_LAYER)

        self.init_param()

    def init_param(self):
        #print('!!!!!!!!!!!!!!!!')
        for name, param in self.backbone.named_parameters():
            #print(name)
            if 'large' not in self.model_cfg.ENCODER:

                if 'layer.11' not in name and 'layer.10' not in name and 'layer.9' not in name and 'layer.8' not in name:
                    param.requires_grad = False
                else:
                    #print('????????')
                    print(name + ' need grads')
                    param.requires_grad = True
            else:
                if 'layer.21' not in name and 'layer.22' not in name and 'layer.23' not in name and 'layer.20' not in name: #  and 'layer.9' not in name
                    param.requires_grad = False
                else:
                    #print('????????')
                    print(name + ' need grads')
                    param.requires_grad = True
        sys.stdout.flush()
        

    def forward(self, textFea, maskTexts):

        textFea = self.backbone(textFea)

        textFea = F.normalize(textFea, p=2, dim=-1)
        # print(textFea.shape) # torch.Size([75, 80, 1024])
        # print(maskTexts.shape)
        # print(1)
        textFea = self.learnable['textAtt'](textFea.transpose(0, 1), src_key_padding_mask=(maskTexts == 0)).transpose(0,1)
        # print(textFea.shape) # torch.Size([75, 80, 1024])
        # print(2)
        tmpMask = torch.where(maskTexts == 1, torch.tensor([1.0], device=maskTexts.device),
                              torch.tensor([0.0], device=maskTexts.device))
        textFea = (textFea * tmpMask.unsqueeze(-1)).sum(dim=1) / tmpMask.sum(dim=1).unsqueeze(-1)  # (bs, dim)
        # print(textFea.shape) # torch.Size([75, 80, 1024])
        # print(3)
        textFea = self.learnable['textFC'](textFea)
        # print(textFea.shape) # torch.Size([75, 80, 1024])
        # print(4)
        return textFea



class VL_model(nn.Module):

    def __init__(self, model_cfg):
        super(VL_model, self).__init__()

        self.model_cfg = model_cfg

        self.learnable = nn.ModuleDict()
        self.learnable['imgencoder'] = ImgLearnableEncoder(model_cfg)
        self.learnable['imgencoder_mom'] = ImgLearnableEncoder(model_cfg)
        self.learnable['textencoder'] = TextLearnableEncoder(model_cfg)
        self.learnable['textencoder_mom'] = TextLearnableEncoder(model_cfg)
        #self.generator = Generator(model_cfg)

        ############ add new params in .yml config file
        self.K = model_cfg.QUEUE_SIZE    # 6400
        self.m = model_cfg.MOMENTUM      # 0.9
        self.T = model_cfg.TEMPERATURE   # 0.07
        self.topk = model_cfg.TOPK       # 5
        self.multi_label = False
        ############ add new params in .yml config file

        # init the parameter of two models
        self.init_param() 
        # create the img queue 
        self.register_buffer("img_queue", torch.randn(model_cfg.IMG_FEATURE_DIM, self.K))
        self.img_queue = nn.functional.normalize(self.img_queue, dim=0)
        self.register_buffer("img_queue_ptr", torch.zeros(1, dtype=torch.long)) # image queue points
        # create the text queue
        self.register_buffer("text_queue", torch.randn(model_cfg.IMG_FEATURE_DIM, self.K))
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)
        self.register_buffer("text_queue_ptr", torch.zeros(1, dtype=torch.long)) # text queue points


    def init_param(self):

        for param_q, param_k in zip(self.learnable['imgencoder'].parameters(), self.learnable['imgencoder_mom'].parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(self.learnable['textencoder'].parameters(), self.learnable['textencoder_mom'].parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder for image modal
        """
        for param_q, param_k in zip(self.learnable['imgencoder'].parameters(), self.learnable['imgencoder_mom'].parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.learnable['textencoder'].parameters(), self.learnable['textencoder_mom'].parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, option='img'):
        # option in 
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        if option == 'img':
            ptr = int(self.img_queue_ptr)
            assert self.K % batch_size == 0  # for simplicity

            # replace the keys at ptr (dequeue and enqueue)
            self.img_queue[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.K  # move pointer

            self.img_queue_ptr[0] = ptr

        else:

            ptr = int(self.text_queue_ptr)
            assert self.K % batch_size == 0  # for simplicity

            # replace the keys at ptr (dequeue and enqueue)
            self.text_queue[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.K  # move pointer

            self.text_queue_ptr[0] = ptr


    @torch.no_grad()
    def _batch_shuffle_ddp(self, x, x_mask):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        x_mask_gather = concat_all_gather(x_mask)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], x_mask_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, x_mask, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        x_mask_gather = concat_all_gather(x_mask)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], x_mask_gather[idx_this]


    def forward(self, imgFea, texts, maskImages, maskTexts, text_lens, image_boxs, is_training=True):

        if self.model_cfg.IS_EXTRACT:
            return self.extract(imgFea, texts, maskImages, maskTexts, image_boxs)

        batch_size = imgFea.size(0)

        imgFea_q = self.learnable['imgencoder'](imgFea, maskImages, image_boxs) # <bsz, img_dim>
        imgFea_q = F.normalize(imgFea_q, p=2, dim=-1)
        textFea_q = self.learnable['textencoder'](texts, maskTexts) # <bsz, img_dim>
        textFea_q = F.normalize(textFea_q, p=2, dim=-1)
        
        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            imgFea, image_boxs, idx_unshuffle = self._batch_shuffle_ddp(imgFea, image_boxs)

            imgFea_k = self.learnable['imgencoder_mom'](imgFea, maskImages, image_boxs) # <bsz, img_dim>
            imgFea_k = F.normalize(imgFea_k, p=2, dim=-1)

            # undo shuffle
            imgFea_k, image_boxs = self._batch_unshuffle_ddp(imgFea_k, image_boxs, idx_unshuffle)

            # shuffle for making use of BN
            texts, maskTexts, idx_unshuffle = self._batch_shuffle_ddp(texts, maskTexts)

            textFea_k = self.learnable['textencoder_mom'](texts, maskTexts) # <bsz, img_dim>
            textFea_k = F.normalize(textFea_k, p=2, dim=-1)

            # undo shuffle
            textFea_k, maskTexts = self._batch_unshuffle_ddp(textFea_k, maskTexts, idx_unshuffle)


        # compute logits for image -> text
        # positive logits: Nx1
        i2t_l_pos = torch.einsum('nc,nc->n', [imgFea_q, textFea_k]).unsqueeze(-1)
        # negative logits: NxK
        i2t_l_neg = torch.einsum('nc,ck->nk', [imgFea_q, self.text_queue.clone().detach()])

        # logits: Nx(1+K)
        i2t_logits = torch.cat([i2t_l_pos, i2t_l_neg], dim=-1)
        i2t_logits /= self.T

        # compute logits for text -> image
        # positive logits: Nx1
        t2i_l_pos = torch.einsum('nc,nc->n', [textFea_q, imgFea_k]).unsqueeze(-1)
        # negative logits: NxK
        t2i_l_neg = torch.einsum('nc,ck->nk', [textFea_q, self.img_queue.clone().detach()])

        # logits: Nx(1+K)
        t2i_logits = torch.cat([t2i_l_pos, t2i_l_neg], dim=-1)
        t2i_logits /= self.T


        ### multi-label
        mask = torch.zeros((batch_size, self.K)).bool().cuda()                                # <B, K>

        if self.multi_label:
            mask_sim_txt = textFea_k.matmul(self.text_queue.clone().detach()) # <B, dim>  <dim, K> -> <B, K>
            mask_sim_img = imgFea_k.matmul(self.img_queue.clone().detach())

            _, topkidx_txt = torch.topk(mask_sim_txt, self.topk, dim=1)             # <B, topk>
            _, topkidx_img = torch.topk(mask_sim_img, self.topk, dim=1)             # <B, topk>
            topk_onehot_txt = torch.zeros_like(mask_sim_txt)            # <B, K>
            topk_onehot_txt.scatter_(1, topkidx_txt, 1)                 # one hot vector
            topk_onehot_img = torch.zeros_like(mask_sim_img)            # <B, K> 
            topk_onehot_img.scatter_(1, topkidx_img, 1)                 # one hot vector

            mask[topk_onehot_txt.bool() & topk_onehot_img.bool()] = True # <B, K>


        mask = torch.cat([torch.ones((batch_size, 1), dtype=torch.long, device=mask.device).bool(),
                          mask], dim=1)                                                                  # <B, K+1>

        ### multi-label
        t2i_loss = -1 * F.log_softmax(t2i_logits, dim=1)                                    # <B, 1+K>
        t2i_loss = torch.masked_select(t2i_loss, mask).sum() / batch_size              # masked_select return 1-d tensor
        i2t_loss = -1 * F.log_softmax(i2t_logits, dim=1)
        i2t_loss = torch.masked_select(i2t_loss, mask).sum() / batch_size              # masked_select return 1-d tensor

        loss = t2i_loss + i2t_loss

        ## enqueue and dequeue
        self._dequeue_and_enqueue(imgFea_k, option='img')
        self._dequeue_and_enqueue(textFea_k, option='text')

        # ----------caption-------------
        # TODO: update
        '''
        if is_training:
            caption = None
            caption_loss = self.generator(imgFea_q, texts, text_lens, maskTexts, is_training)
        else:
            caption_loss, caption = self.generator(imgFea_q, texts, text_lens, maskTexts, is_training)
        '''

        return loss#, caption_loss, caption


    def extract(self, imgFea, texts, maskImages, maskTexts, image_boxs):


        imgFea = self.learnable['imgencoder'](imgFea, maskImages, image_boxs) # <bsz, img_dim>
        textFea = self.learnable['textencoder'](texts, maskTexts) # <bsz, img_dim>
        
        imgFea = F.normalize(imgFea, p=2, dim=-1)
        textFea = F.normalize(textFea, p=2, dim=-1)

        retrieval_feat_group = {}

        retrieval_feat_group['img_text'] = (imgFea, textFea)

        return retrieval_feat_group


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output