


import types

from mmdet3d.models import build_model

from mmcv.runner import auto_fp16, force_fp32
from mmcv.runner import load_checkpoint
from mmcv import Config
from mmdet3d.utils import recursive_eval
from torchpack.utils.config import configs

import torch
import torch.nn.functional as F
from mmdet.core import (
    multi_apply,
)

def forward_single_head(self, inputs, img_inputs, metas):
    """Forward function for CenterPoint.
    Args:
        inputs (torch.Tensor): Input feature map with the shape of
            [B, 512, 128(H), 128(W)]. (consistent with L748)
    Returns:
        list[dict]: Output results for tasks.
    """
    batch_size = inputs.shape[0]
    lidar_feat = self.shared_conv(inputs)

    #################################
    # image to BEV
    #################################
    lidar_feat_flatten = lidar_feat.view(
        batch_size, lidar_feat.shape[1], -1
    )  # [BS, C, H*W]
    bev_pos = self.bev_pos.repeat(batch_size, 1, 1).to(lidar_feat.device)

    #################################
    # image guided query initialization
    #################################
    dense_heatmap = self.heatmap_head(lidar_feat)
    dense_heatmap_img = None
    heatmap = dense_heatmap.detach().sigmoid()
    padding = self.nms_kernel_size // 2
    local_max = torch.zeros_like(heatmap)
    # equals to nms radius = voxel_size * out_size_factor * kenel_size
    local_max_inner = F.max_pool2d(
        heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=0
    )
    local_max[:, :, padding:(-padding), padding:(-padding)] = local_max_inner
    ## for Pedestrian & Traffic_cone in nuScenes
    if self.test_cfg["dataset"] == "nuScenes":
        local_max[
            :,
            8,
        ] = F.max_pool2d(heatmap[:, 8], kernel_size=1, stride=1, padding=0)
        local_max[
            :,
            9,
        ] = F.max_pool2d(heatmap[:, 9], kernel_size=1, stride=1, padding=0)
    elif self.test_cfg["dataset"] == "Waymo":  # for Pedestrian & Cyclist in Waymo
        local_max[
            :,
            1,
        ] = F.max_pool2d(heatmap[:, 1], kernel_size=1, stride=1, padding=0)
        local_max[
            :,
            2,
        ] = F.max_pool2d(heatmap[:, 2], kernel_size=1, stride=1, padding=0)
    heatmap = heatmap * (heatmap == local_max)
    heatmap = heatmap.view(batch_size, heatmap.shape[1], -1)

    # top #num_proposals among all classes
    top_proposals = heatmap.view(batch_size, -1).argsort(dim=-1, descending=True)[
        ..., : self.num_proposals
    ]
    top_proposals_class = top_proposals // heatmap.shape[-1]
    top_proposals_index = top_proposals % heatmap.shape[-1]
    query_feat = lidar_feat_flatten.gather(
        index=top_proposals_index[:, None, :].expand(
            -1, lidar_feat_flatten.shape[1], -1
        ),
        dim=-1,
    )
    self.query_labels = top_proposals_class

    # add category embedding
    one_hot = F.one_hot(top_proposals_class, num_classes=self.num_classes).permute(
        0, 2, 1
    )
    query_cat_encoding = self.class_encoding(one_hot.float())
    query_feat += query_cat_encoding

    query_pos = bev_pos.gather(
        index=top_proposals_index[:, None, :]
        .permute(0, 2, 1)
        .expand(-1, -1, bev_pos.shape[-1]),
        dim=1,
    )

    # print('query_feat',query_feat.shape)

    #################################
    # transformer decoder layer (LiDAR feature as K,V)
    #################################
    ret_dicts = []
    for i in range(self.num_decoder_layers):
        prefix = "last_" if (i == self.num_decoder_layers - 1) else f"{i}head_"

        # Transformer Decoder Layer
        # :param query: B C Pq    :param query_pos: B Pq 3/6
        query_feat = self.decoder[i](
            query_feat, lidar_feat_flatten, query_pos, bev_pos
        )
        # print(i,'query_feat2',query_feat.shape)

        # Prediction
        res_layer = self.prediction_heads[i](query_feat)
        res_layer["center"] = res_layer["center"] + query_pos.permute(0, 2, 1)
        first_res_layer = res_layer
        # print('transformer decoder layer (LiDAR feature as K,V)',{k:v.shape for k,v in res_layer.items()})
        ret_dicts.append(res_layer)

        # for next level positional embedding
        query_pos = res_layer["center"].detach().clone().permute(0, 2, 1)

    #################################
    # transformer decoder layer (img feature as K,V)
    #################################
    ret_dicts[0]["query_heatmap_score"] = heatmap.gather(
        index=top_proposals_index[:, None, :].expand(-1, self.num_classes, -1),
        dim=-1,
    )  # [bs, num_classes, num_proposals]
    ret_dicts[0]["dense_heatmap"] = dense_heatmap

    if self.auxiliary is False:
        # only return the results of last decoder layer
        return [ret_dicts[-1]]

    # return all the layer's results for auxiliary superivison
    new_res = {}
    for key in ret_dicts[0].keys():
        # print(key)
        if key not in ["dense_heatmap", "dense_heatmap_old", "query_heatmap_score"]:
            new_res[key] = torch.cat(
                [ret_dict[key] for ret_dict in ret_dicts], dim=-1
            )
        else:
            new_res[key] = ret_dicts[0][key]

    # print(new_res)
    # print({k:v.shape for k,v in new_res.items()})
    # exit(0)
    return [new_res], query_feat

@auto_fp16(apply_to=("img", "points"))
def forward_single(
    self,
    img,
    points,
    camera2ego,
    lidar2ego,
    lidar2camera,
    lidar2image,
    camera_intrinsics,
    camera2lidar,
    img_aug_matrix,
    lidar_aug_matrix,
    metas,
    gt_masks_bev=None,
    gt_bboxes_3d=None,
    gt_labels_3d=None,
    **kwargs,
):
    features = []
    for sensor in (
        self.encoders if self.training else list(self.encoders.keys())[::-1]
    ):
        if sensor == "camera":
            feature = self.extract_camera_features(
                img,
                points,
                camera2ego,
                lidar2ego,
                lidar2camera,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                metas,
            )
        elif sensor == "lidar":
            feature = self.extract_lidar_features(points)
        else:
            raise ValueError(f"unsupported sensor: {sensor}")
        features.append(feature)

    if not self.training:
        # avoid OOM
        features = features[::-1]

    if self.fuser is not None:
        x = self.fuser(features)
    else:
        assert len(features) == 1, features
        x = features[0]

    batch_size = x.shape[0]

    x = self.decoder["backbone"](x)
    x = self.decoder["neck"](x)

    if self.training:
        outputs = {}
        for type, head in self.heads.items():
            if type == "object":
                pred_dict, queries = head(x, metas)
                losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
            elif type == "map":
                losses = head(x, gt_masks_bev)
            else:
                raise ValueError(f"unsupported head: {type}")
            for name, val in losses.items():
                if val.requires_grad:
                    outputs[f"loss/{type}/{name}"] = val * self.loss_scale[type]
                else:
                    outputs[f"stats/{type}/{name}"] = val
        return outputs, x
    else:
        outputs = [{} for _ in range(batch_size)]
        for type, head in self.heads.items():
            if type == "object":
                pred_dict, queries = head(x, metas)
                bboxes = head.get_bboxes(pred_dict, metas)
                for k, (boxes, scores, labels) in enumerate(bboxes):
                    outputs[k].update(
                        {
                            "boxes_3d": boxes,#.to("cpu"),
                            "scores_3d": scores,#.cpu(),
                            "labels_3d": labels,#.cpu(),
                        }
                    )
            elif type == "map":
                logits = head(x)
                for k in range(batch_size):
                    outputs[k].update(
                        {
                            "masks_bev": logits[k].cpu(),
                            "gt_masks_bev": gt_masks_bev[k].cpu(),
                        }
                    )
            else:
                raise ValueError(f"unsupported head: {type}")
        return outputs, x, queries


def forward_detector_tracking(self,*args,**kwargs):
    bbox_list, pts_feats, queries = self(*args,**kwargs)
    # print(out)
    # print(feats)
    # print([x.shape for x in feats])
    # exit(0)
    losses = {}
    # img_feats = None
    before_neck = None
    before_backbone = None
    middle_feats = None
    return losses, bbox_list, queries, pts_feats, before_neck, before_backbone, middle_feats

def load_pretrained_model(config, loadpath, test=False, cfg_type='torchpack'):

    if cfg_type == 'mmcv':
        cfg = Config.fromfile(config)
    elif cfg_type == 'torchpack':
        configs.load(config, recursive=True)
        cfg = Config(recursive_eval(configs), filename=config)

    # print(cfg.model.test_cfg)
    if test == True:
        model = build_model(cfg.model)
    else:
        model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    ckpt = load_checkpoint(
        model=model,
        filename=loadpath
    )
    return model

def forward_head(self, feats, metas):
    """Forward pass.
    Args:
        feats (list[torch.Tensor]): Multi-level features, e.g.,
            features produced by FPN.
    Returns:
        tuple(list[dict]): Output results. first index by level, second index by layer
    """
    if isinstance(feats, torch.Tensor):
        feats = [feats]
    res = multi_apply(self.forward_single, feats, [None], [metas])
    # assert len(res) == 1, "only support one level features."
    return res


def load_pretrained_detector(config,loadpath,test=False,cfg_type='torchpack'):
    """ Load pretrained cetnerpoint and modify its runtime behaviour"""
    detector = load_pretrained_model(config=config,loadpath=loadpath,test=test,cfg_type=cfg_type)

    detector.heads['object'].forward_single = types.MethodType(forward_single_head,detector.heads['object'])
    detector.heads['object'].forward = types.MethodType(forward_head,detector.heads['object'])
    detector.forward_single = types.MethodType(forward_single,detector)
    # centerpoint.forward_pts_train = types.MethodType(forward_pts_train,centerpoint)
    # centerpoint.forward_img_train = types.MethodType(forward_img_train,centerpoint)
    # centerpoint.simple_test_pts = types.MethodType(simple_test_pts,centerpoint)
    # centerpoint.simple_test = types.MethodType(simple_test,centerpoint)
    # centerpoint.extract_pts_feat = types.MethodType(extract_pts_feat,centerpoint)
    # centerpoint.extract_feat = types.MethodType(extract_feat,centerpoint)
    detector.forward_detector_tracking = types.MethodType(forward_detector_tracking,detector)

    # centerpoint.pts_bbox_head.get_bboxes = types.MethodType(
    #         get_bboxes,centerpoint.pts_bbox_head)
    # centerpoint.pts_bbox_head.bbox_coder.decode = types.MethodType(
    #         decode,centerpoint.pts_bbox_head.bbox_coder)

    return detector