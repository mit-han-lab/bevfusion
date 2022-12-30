


import types
import torch

from mmdet3d.models import build_model

from mmcv.runner import auto_fp16, force_fp32
from mmcv.runner import load_checkpoint
from mmcv import Config
from mmdet3d.utils import recursive_eval
from torchpack.utils.config import configs




# def forward_detector_tracking(self,points=None,img_metas=None,gt_bboxes_3d=None,
#                     gt_labels_3d=None,gt_labels=None,gt_bboxes=None,img=None,
#                     proposals=None,gt_bboxes_ignore=None, compute_loss=True):

#     losses, img_feats, (pts_feats, before_neck, before_backbone, middle_feats,), pts_bbox_outs = self(
#                 points=points, img_metas=img_metas, gt_bboxes_3d=gt_bboxes_3d,
#                 gt_labels_3d=gt_labels_3d, gt_labels=gt_labels, gt_bboxes=gt_bboxes,
#                 img=img, proposals=proposals, gt_bboxes_ignore=gt_bboxes_ignore,
#                 compute_loss=compute_loss
#             )

#     bbox_list = self.pts_bbox_head.get_bboxes(
#             pts_bbox_outs, img_metas, rescale=False)

#     return losses, bbox_list, img_feats, pts_feats, before_neck, before_backbone, middle_feats 


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
                pred_dict = head(x, metas)
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
                pred_dict = head(x, metas)
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
        return outputs, x


def forward_detector_tracking(self,*args,**kwargs):
    bbox_list, pts_feats = self(*args,**kwargs)
    # print(out)
    # print(feats)
    # print([x.shape for x in feats])
    # exit(0)
    losses = {}
    img_feats = None
    before_neck = None
    before_backbone = None
    middle_feats = None
    return losses, bbox_list, img_feats, pts_feats, before_neck, before_backbone, middle_feats

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


def load_pretrained_detector(config,loadpath,test=False,cfg_type='torchpack'):
    """ Load pretrained cetnerpoint and modify its runtime behaviour"""
    detector = load_pretrained_model(config=config,loadpath=loadpath,test=test,cfg_type=cfg_type)


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