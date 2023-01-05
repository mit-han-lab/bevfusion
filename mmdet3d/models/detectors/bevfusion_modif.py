


import types

from mmdet3d.models import build_model

from mmcv.runner import auto_fp16, force_fp32
from mmcv.runner import load_checkpoint
from mmcv import Config
from mmdet3d.utils import recursive_eval
from torchpack.utils.config import configs

from mmdet3d.ops.iou3d.iou3d_utils import nms_gpu
import torch
import torch.nn.functional as F
from mmdet3d.core import (
    PseudoSampler,
    circle_nms,
    draw_heatmap_gaussian,
    gaussian_radius,
    xywhr2xyxyr,
)

from mmdet.core import (
    AssignResult,
    build_assigner,
    build_bbox_coder,
    build_sampler,
    multi_apply,
)

# modify decode to return masked queries
def decode(self, heatmap, rot, dim, center, height, vel, filter=False, queries=None):
    """Decode bboxes.
    Args:
        heat (torch.Tensor): Heatmap with the shape of [B, num_cls, num_proposals].
        rot (torch.Tensor): Rotation with the shape of
            [B, 1, num_proposals].
        dim (torch.Tensor): Dim of the boxes with the shape of
            [B, 3, num_proposals].
        center (torch.Tensor): bev center of the boxes with the shape of
            [B, 2, num_proposals]. (in feature map metric)
        hieght (torch.Tensor): height of the boxes with the shape of
            [B, 2, num_proposals]. (in real world metric)
        vel (torch.Tensor): Velocity with the shape of [B, 2, num_proposals].
        filter: if False, return all box without checking score and center_range
    Returns:
        list[dict]: Decoded boxes.
    """
    # class label
    final_preds = heatmap.max(1, keepdims=False).indices
    final_scores = heatmap.max(1, keepdims=False).values

    # change size to real world metric
    center[:, 0, :] = center[:, 0, :] * self.out_size_factor * self.voxel_size[0] + self.pc_range[0]
    center[:, 1, :] = center[:, 1, :] * self.out_size_factor * self.voxel_size[1] + self.pc_range[1]
    # center[:, 2, :] = center[:, 2, :] * (self.post_center_range[5] - self.post_center_range[2]) + self.post_center_range[2]
    dim[:, 0, :] = dim[:, 0, :].exp()
    dim[:, 1, :] = dim[:, 1, :].exp()
    dim[:, 2, :] = dim[:, 2, :].exp()
    height = height - dim[:, 2:3, :] * 0.5  # gravity center to bottom center
    rots, rotc = rot[:, 0:1, :], rot[:, 1:2, :]
    rot = torch.atan2(rots, rotc)

    if vel is None:
        final_box_preds = torch.cat([center, height, dim, rot], dim=1).permute(0, 2, 1)
    else:
        final_box_preds = torch.cat([center, height, dim, rot, vel], dim=1).permute(0, 2, 1)

    predictions_dicts = []
    for i in range(heatmap.shape[0]):
        boxes3d = final_box_preds[i]
        scores = final_scores[i]
        labels = final_preds[i]
        predictions_dict = {
            'bboxes': boxes3d,
            'scores': scores,
            'labels': labels
        }
        predictions_dicts.append(predictions_dict)

    if filter is False:
        return predictions_dicts

    # use score threshold
    if self.score_threshold is not None:
        thresh_mask = final_scores > self.score_threshold

    if self.post_center_range is not None:
        self.post_center_range = torch.tensor(
            self.post_center_range, device=heatmap.device)
        mask = (final_box_preds[..., :3] >=
                self.post_center_range[:3]).all(2)
        mask &= (final_box_preds[..., :3] <=
                    self.post_center_range[3:]).all(2)

        predictions_dicts = []
        for i in range(heatmap.shape[0]):
            cmask = mask[i, :]
            if self.score_threshold:
                cmask &= thresh_mask[i]

            boxes3d = final_box_preds[i, cmask]
            scores = final_scores[i, cmask]
            labels = final_preds[i, cmask]
            predictions_dict = {
                'bboxes': boxes3d,
                'scores': scores,
                'labels': labels
            }

            if queries is not None:
                # print('in decode',queries[i].shape)
                queries[i] = queries[i][:,:,cmask]
                # print('in decode after',queries[i].shape)

            predictions_dicts.append(predictions_dict)
    else:
        raise NotImplementedError(
            'Need to reorganize output as a batch, only '
            'support post_center_range is not None for now!')

    if queries is not None:
        return predictions_dicts, queries
    else:
        return predictions_dicts


def get_bboxes_head(self, preds_dicts, metas, img=None, rescale=False, for_roi=False, queries=None):
    """Generate bboxes from bbox head predictions.
    Args:
        preds_dicts (tuple[list[dict]]): Prediction results.
    Returns:
        list[list[dict]]: Decoded bbox, scores and labels for each layer & each batch
    """
    rets = []
    for layer_id, preds_dict in enumerate(preds_dicts):
        batch_size = preds_dict[0]["heatmap"].shape[0]
        batch_score = preds_dict[0]["heatmap"][..., -self.num_proposals :].sigmoid()
        # if self.loss_iou.loss_weight != 0:
        #    batch_score = torch.sqrt(batch_score * preds_dict[0]['iou'][..., -self.num_proposals:].sigmoid())
        one_hot = F.one_hot(
            self.query_labels, num_classes=self.num_classes
        ).permute(0, 2, 1)
        batch_score = batch_score * preds_dict[0]["query_heatmap_score"] * one_hot

        batch_center = preds_dict[0]["center"][..., -self.num_proposals :]
        batch_height = preds_dict[0]["height"][..., -self.num_proposals :]
        batch_dim = preds_dict[0]["dim"][..., -self.num_proposals :]
        batch_rot = preds_dict[0]["rot"][..., -self.num_proposals :]
        batch_vel = None
        if "vel" in preds_dict[0]:
            batch_vel = preds_dict[0]["vel"][..., -self.num_proposals :]

        # print('in get bboxes',batch_score.shape)
        temp = self.bbox_coder.decode(
            batch_score,
            batch_rot,
            batch_dim,
            batch_center,
            batch_height,
            batch_vel,
            filter=True,
            queries=queries,
        )

        if queries is not None:
            temp, queries = temp


        if self.test_cfg["dataset"] == "nuScenes":
            self.tasks = [
                dict(
                    num_class=8,
                    class_names=[],
                    indices=[0, 1, 2, 3, 4, 5, 6, 7],
                    radius=-1,
                ),
                dict(
                    num_class=1,
                    class_names=["pedestrian"],
                    indices=[8],
                    radius=0.175,
                ),
                dict(
                    num_class=1,
                    class_names=["traffic_cone"],
                    indices=[9],
                    radius=0.175,
                ),
            ]
        elif self.test_cfg["dataset"] == "Waymo":
            self.tasks = [
                dict(num_class=1, class_names=["Car"], indices=[0], radius=0.7),
                dict(
                    num_class=1, class_names=["Pedestrian"], indices=[1], radius=0.7
                ),
                dict(num_class=1, class_names=["Cyclist"], indices=[2], radius=0.7),
            ]

        ret_layer = []
        for i in range(batch_size):
            boxes3d = temp[i]["bboxes"]
            scores = temp[i]["scores"]
            labels = temp[i]["labels"]
            # print('in get bboxes bboxes shape:',boxes3d.shape)
            ## adopt circle nms for different categories
            if self.test_cfg["nms_type"] != None:
                keep_mask = torch.zeros_like(scores)
                for task in self.tasks:
                    task_mask = torch.zeros_like(scores)
                    for cls_idx in task["indices"]:
                        task_mask += labels == cls_idx
                    task_mask = task_mask.bool()
                    if task["radius"] > 0:
                        if self.test_cfg["nms_type"] == "circle":
                            boxes_for_nms = torch.cat(
                                [
                                    boxes3d[task_mask][:, :2],
                                    scores[:, None][task_mask],
                                ],
                                dim=1,
                            )
                            task_keep_indices = torch.tensor(
                                circle_nms(
                                    boxes_for_nms.detach().cpu().numpy(),
                                    task["radius"],
                                )
                            )
                        else:
                            boxes_for_nms = xywhr2xyxyr(
                                metas[i]["box_type_3d"](
                                    boxes3d[task_mask][:, :7], 7
                                ).bev
                            )
                            top_scores = scores[task_mask]
                            task_keep_indices = nms_gpu(
                                boxes_for_nms,
                                top_scores,
                                thresh=task["radius"],
                                pre_maxsize=self.test_cfg["pre_maxsize"],
                                post_max_size=self.test_cfg["post_maxsize"],
                            )
                    else:
                        task_keep_indices = torch.arange(task_mask.sum())
                        
                    if task_keep_indices.shape[0] != 0:
                        keep_indices = torch.where(task_mask != 0)[0][
                            task_keep_indices
                        ]
                        keep_mask[keep_indices] = 1


                keep_mask = keep_mask.bool()
                ret = dict(
                    bboxes=boxes3d[keep_mask],
                    scores=scores[keep_mask],
                    labels=labels[keep_mask],
                )
                if queries is not None:
                    # print('getbboxes_shapes',queries[i].shape, keep_mask.shape)
                    queries[i] = queries[i][:,:,keep_mask]
                
            else:  # no nms
                ret = dict(bboxes=boxes3d, scores=scores, labels=labels)
            ret_layer.append(ret)
        rets.append(ret_layer)
    assert len(rets) == 1
    assert len(rets[0]) == 1
    res = [
        [
            metas[0]["box_type_3d"](
                rets[0][0]["bboxes"], box_dim=rets[0][0]["bboxes"].shape[-1]
            ),
            rets[0][0]["scores"],
            rets[0][0]["labels"].int(),
        ]
    ]

    if queries is not None:
        return res, queries
    else:
        return res

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
                # print(pred_dict)
                # print()
                # print('before get_bboxes',[{k:v.shape for k,v in d.items()} for d in pred_dict[0]])
                bboxes, queries = head.get_bboxes(pred_dict, metas, queries=queries)
                # print('after get_bboxes',[(boxes.tensor.shape, scores.shape, labels.shape) for boxes, scores, labels in bboxes])
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
    
    detector.heads['object'].bbox_coder.decode = types.MethodType(decode,detector.heads['object'].bbox_coder)
    detector.heads['object'].get_bboxes = types.MethodType(get_bboxes_head,detector.heads['object'])
    detector.heads['object'].forward_single = types.MethodType(forward_single_head,detector.heads['object'])
    detector.heads['object'].forward = types.MethodType(forward_head,detector.heads['object'])
    detector.forward_single = types.MethodType(forward_single,detector)
    detector.forward_detector_tracking = types.MethodType(forward_detector_tracking,detector)

    return detector