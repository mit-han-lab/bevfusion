import torch.nn as nn
from mmdet.models import DETECTORS
from .tracking_helpers import interpolate_bev_2d
from torch.nn.modules.transformer import (TransformerEncoder, TransformerDecoder, 
                                          TransformerDecoderLayer, TransformerEncoderLayer,)
from .lanegcn_nets import PostRes,LinearRes
from .loftr import LocalFeatureTransformer
from .backbone_net import Pointnet_Backbone
# from .node_pooling import GatedPooling
# from torch_geometric.nn.conv import GATv2Conv,GatedGraphConv
# from torch_geometric.data import Data

from mmdet.models import BaseDetector
import torch.distributed as dist
# from pytorch3d.loss import chamfer_distance

import torch
import copy
import time 

from .attention import corss_attention,local_self_attention

module_obj = {
    'Linear':nn.Linear,
    'ReLU':nn.ReLU,
    'LSTM':nn.LSTM,
    'GroupNorm':nn.GroupNorm,
    'Embedding':nn.Embedding,
    'MultiheadAttention':nn.MultiheadAttention,
    'TransformerEncoder':TransformerEncoder,
    'TransformerEncoderLayer':TransformerEncoderLayer,
    'TransformerDecoder':TransformerDecoder,
    'TransformerDecoderLayer':TransformerDecoderLayer,
    'LayerNorm':nn.LayerNorm,
    'PostRes':PostRes,
    'LinearRes':LinearRes,
    'LocalFeatureTransformer':LocalFeatureTransformer,
    # 'GatedPooling':GatedPooling,
    # 'GATv2Conv':GATv2Conv,
    # 'GatedGraphConv':GatedGraphConv,
    'Pointnet_Backbone':Pointnet_Backbone,
    'corss_attention':corss_attention,
    'local_self_attention':local_self_attention,
    'Conv1d':nn.Conv1d,
    'Conv2d':nn.Conv2d,
    'BatchNorm1d':nn.BatchNorm1d,
    'Sigmoid':nn.Sigmoid,
}

def build_module(cfg):
    if cfg == None or cfg == {}:
        return None

    if isinstance(cfg, list):
        return build_sequential(cfg)

    cls_ = module_obj[cfg['type']]
    del cfg['type']
    return cls_(**cfg)


def build_sequential(module_list):
    if module_list == None or module_list == {}:
        return None
        
    modules = []
    for cfg in module_list:
        modules.append(build_module(cfg))
    return nn.Sequential(*modules)

def build_decisions(decisions):
    if decisions == None or decisions == {}:
        return None

    emb = decisions['embedding']
    return nn.ModuleDict({k:build_module(copy.deepcopy(emb)) for k in decisions if k != 'embedding'})

def get_accuracy(y_true, y_prob):
    assert y_true.ndim == 1 and y_true.size() == y_prob.size()
    y_prob = y_prob > 0.5
    return (y_true == y_prob).sum().item() / y_true.size(0)


@DETECTORS.register_module()
class ReIDNet(BaseDetector):
    def __init__(self,hidden_size,backbone,cls_head,match_head,shape_head,
                 cross_stage1,local_stage1,cross_stage2,local_stage2,
                 compute_summary=True,train_cfg=None,test_cfg=None):
        super().__init__()
        print('Entering ReIDNet')
        self.hidden_size = hidden_size
        self.backbone = build_module(backbone)
        self.cls_head = build_module(cls_head)
        self.match_head = build_module(match_head)
        self.shape_head = build_module(shape_head)
        
        self.cross_stage1 = build_module(cross_stage1)                 
        self.local_stage1 = build_module(local_stage1)

        self.cross_stage2 = build_module(cross_stage2)          
        self.local_stage2 = build_module(local_stage2)

        self.maxpool = nn.MaxPool1d(32)
        self.bce = nn.BCEWithLogitsLoss()
        self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss(log_target=True,reduction='none')
        self.lsmx = nn.LogSoftmax(dim=1)

        self.verbose=False

        self.compute_summary = compute_summary
        print('Exiting ReIDNet')

    

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.

        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        # print(data_batch)
        # exit(0)
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def forward_train(self,sparse_1,sparse_2,dense_1,dense_2,label_1,label_2, match):
        batch = len(sparse_1)
        log_vars = {}
        losses = {}
        sparse_1 = torch.stack(sparse_1,dim=0)
        sparse_2 = torch.stack(sparse_2,dim=0)
        dense_1 = torch.stack(dense_1,dim=0)
        dense_2 = torch.stack(dense_2,dim=0)
        label_1 = torch.cat(label_1,dim=0)
        label_2 = torch.cat(label_2,dim=0)
        match = torch.cat(match,dim=0)
        # print(sparse_1,sparse_2,dense_1,dense_2,label_1,label_2, match)
        # print('sparse_1',sparse_1.shape)
        # print('sparse_2',sparse_2.shape)
        # print('dense_1',dense_1.shape)
        # print('dense_2',dense_2.shape)
        # print('label_1',label_1.shape)
        # print('label_2',label_2.shape)
        # print('match',match.shape)

        xyz1, h1 = self.backbone(sparse_1,[256, 128, 64])
        xyz2, h2 = self.backbone(sparse_2,[256, 128, 64])

        match_in = self.xcorr(h1,xyz1,h2,xyz2)
        # print(match_in)
        # print(match_in.shape)
        match_preds = self.match_head(self.maxpool(match_in.permute(0,2,1)).squeeze(-1)).squeeze(1)
        # print(match_preds.shape)
        match_loss = self.bce(match_preds,match)

        h_stacked = torch.cat([h1,h2],dim=0)

        cls_preds = self.cls_head(self.maxpool(h_stacked.permute(0,2,1)).squeeze(-1)).squeeze(1)
        # print(cls_preds.shape)
        cls_loss = self.ce(cls_preds,torch.cat([label_1,label_2],dim=0))

        shape_preds = self.shape_head(torch.cat([h1,h2],dim=0).permute(0,2,1))
        # print(shape_preds.shape)
        shape_loss,_ = chamfer_distance(shape_preds,torch.cat([dense_1, dense_2],dim=0),)

        # print(h1.reshape(batch,-1))
        # print(h2.reshape(batch,-1))
        
        # print(h1.reshape(batch,-1).log())
        # print(h2.reshape(batch,-1).log())
        kl_loss = self.kl(self.lsmx(h1.reshape(batch,-1)),self.lsmx(h2.reshape(batch,-1))).mean(dim=1)
        # print(kl_loss)
        # exit(0)
        where_no_match = torch.where(match == 0)
        kl_loss[where_no_match] = kl_loss[where_no_match] * -1
        
        kl_loss = kl_loss.mean()

        if self.compute_summary:
            log_vars['match_loss'] = match_loss.item()
            log_vars['shape_loss'] = shape_loss.item()
            log_vars['cls_loss'] = cls_loss.item()
            log_vars['kl_loss'] = kl_loss.item()
            log_vars['match_acc'] = (nn.Sigmoid()(match_preds) > 0.5).float().eq(match).float().mean().item()
            log_vars['cls_acc'] = cls_preds.argmax(dim=1).eq(torch.cat([label_1,label_2],dim=0)).float().mean().item()

            print(log_vars)


        losses['reid_loss'] = match_loss + shape_loss + cls_loss + kl_loss

        return losses, log_vars


    def xcorr(self, search_feat, search_xyz, template_feat, template_xyz):       
        search_feat1_a = self.cross_stage1(search_feat, search_xyz, template_feat, template_xyz)
        search_feat1_b = self.local_stage1(search_feat1_a, search_xyz)
        search_feat2_a = self.cross_stage2(search_feat1_b, search_xyz, template_feat, template_xyz)
        search_feat2_b = self.local_stage2(search_feat2_a, search_xyz)

        return search_feat2_b  


    def forward_test(self,sparse_1,sparse_2,dense_1,dense_2,label_1,label_2, match):
        batch = len(sparse_1)
        log_vars = {}
        losses = {}
        sparse_1 = torch.stack(sparse_1,dim=0)
        sparse_2 = torch.stack(sparse_2,dim=0)
        dense_1 = torch.stack(dense_1,dim=0)
        dense_2 = torch.stack(dense_2,dim=0)
        label_1 = torch.cat(label_1,dim=0)
        label_2 = torch.cat(label_2,dim=0)
        match = torch.cat(match,dim=0)
        # print(sparse_1,sparse_2,dense_1,dense_2,label_1,label_2, match)
        # print('sparse_1',sparse_1.shape)
        # print('sparse_2',sparse_2.shape)
        # print('dense_1',dense_1.shape)
        # print('dense_2',dense_2.shape)
        # print('label_1',label_1.shape)
        # print('label_2',label_2.shape)
        # print('match',match.shape)

        xyz1, h1 = self.backbone(sparse_1,[256, 128, 64])
        xyz2, h2 = self.backbone(sparse_2,[256, 128, 64])

        match_in = self.xcorr(h1,xyz1,h2,xyz2)
        # print(match_in)
        # print(match_in.shape)
        match_preds = self.match_head(self.maxpool(match_in.permute(0,2,1)).squeeze(-1)).squeeze(1)
        print(match_preds.shape)
        match_loss = self.bce(match_preds,match)

        h_stacked = torch.cat([h1,h2],dim=0)

        cls_preds = self.cls_head(self.maxpool(h_stacked.permute(0,2,1)).squeeze(-1)).squeeze(1)
        print(cls_preds.shape)
        cls_loss = self.ce(cls_preds,torch.cat([label_1,label_2],dim=0))

        shape_preds = self.shape_head(torch.cat([h1,h2],dim=0).permute(0,2,1))
        print(shape_preds.shape)
        shape_loss,_ = chamfer_distance(shape_preds,torch.cat([dense_1, dense_2],dim=0),)
        
        
        kl_loss = self.kl(h1.reshape(batch,-1).log(),h2.reshape(batch,-1).log()).mean(dim=1)
        where_no_match = torch.where(match == 0)
        kl_loss[where_no_match] = kl_loss[where_no_match] * -1
        kl_loss = kl_loss.mean()

        if self.compute_summary:
            log_vars['match_loss'] = match_loss.item()
            log_vars['shape_loss'] = shape_loss.item()
            log_vars['cls_loss'] = cls_loss.item()
            log_vars['kl_loss'] = kl_loss.item()


        losses['reid_loss'] = match_loss + shape_loss + cls_loss + kl_loss

        return losses



    #MMDET3d API

    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a
                  weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                  logger.
                - ``num_samples`` indicates the batch size (when the model is
                  DDP, it means the batch size on each GPU), which is used for
                  averaging the logs.
        """
        
        if self.verbose:
            t1 = time.time()
            print("{} Starting train_step()".format(self.log_msg()))


        losses, log_vars_train = self(**data)
        loss, log_vars = self._parse_losses(losses)

        log_vars.update(log_vars_train)
        
        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['sparse_1']))
        
        if self.verbose:
            print("{} Ending train_step() after {}s".format(self.log_msg(),time.time()-t1))

        return outputs




    def val_step(self, data, optimizer=None):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['sparse_1']))

        return outputs


    # def train_step(self, data_batch, optimizer,*args, **kwargs):
    #     return self.forward(data_batch,*args, **kwargs)


    def extract_feat(self,*args,**kwargs):
        raise NotImplementedError

    def show_result(self):
        raise NotImplementedError

    def aug_test(self,*args,**kwargs):#abstract method needs to be re-implemented
        raise NotImplementedError

    def simple_test(self,*args,**kwargs):#abstract method needs to be re-implemented
        raise NotImplementedError

    def init_weights(self):
        pass