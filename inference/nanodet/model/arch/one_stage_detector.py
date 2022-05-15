import time
import torch
import torch.nn as nn
from ..backbone import build_backbone
from ..fpn import build_fpn
from ..head import build_head
import numpy
class pan_segment(nn.Module):
  def __init__ (self, backbone_cfg,fpn_cfg=None):
    super(pan_segment,self).__init__()
    self.backbone = build_backbone(backbone_cfg)
    if fpn_cfg is not None:
        self.fpn = build_fpn(fpn_cfg)
  def forward(self, x):
    x = self.backbone(x)
    if hasattr(self, 'fpn'):
       x,out_segment= self.fpn(x)
    return out_segment
class OneStageDetector(nn.Module):
    def __init__(self,
                 backbone_cfg,
                 fpn_cfg=None,
                 head_cfg=None,):
        super(OneStageDetector, self).__init__()
        self.backbone = build_backbone(backbone_cfg)
        if fpn_cfg is not None:
            self.fpn = build_fpn(fpn_cfg)
            self.task = fpn_cfg.pop('task')
        if head_cfg is not None:
            self.head = build_head(head_cfg)
    def forward(self, x):
        x = self.backbone(x)
        if self.task == "multi":
            if hasattr(self, 'fpn'):
                x,out_segment= self.fpn(x)
            if hasattr(self, 'head'):
                x = self.head(x)
            return x,out_segment
        elif self.task == "segment":
            if hasattr(self, 'fpn'):
                out_segment= self.fpn(x)
            return out_segment
        else:
            if hasattr(self, 'fpn'):
                x= self.fpn(x)
            if hasattr(self, 'head'):
                #print('x',x.shape)
                x = self.head(x)
            return x
    def inference(self, meta):
        with torch.no_grad():
            if self.task == "multi":
                #torch.cuda.synchronize()
                preds,preds_segment = self(meta['img'])
                #torch.cuda.synchronize()
                results = self.head.post_process(preds, meta)
                #torch.cuda.synchronize()
                return results,preds_segment
            elif self.task == "segment":
                #torch.cuda.synchronize()
                preds_segment = self(meta['img'])
                print(preds_segment.shape)
                #torch.cuda.synchronize()
                #torch.cuda.synchronize()
                return preds_segment
            else:
                #torch.cuda.synchronize()
                preds = self(meta['img'])
                #torch.cuda.synchronize()
                results = self.head.post_process(preds, meta)
                #torch.cuda.synchronize()
                return results


    def forward_train(self, gt_meta):
        preds,preds_segment = self(gt_meta['img'])
        loss, loss_states =self.head.loss(preds,preds_segment, gt_meta)
        return preds, loss, loss_states,preds_segment,gt_meta
