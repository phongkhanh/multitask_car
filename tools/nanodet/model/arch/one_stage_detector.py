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
        #self.segment =pan_segment(backbone_cfg,fpn_cfg)
    def forward(self, x):
        #print('self.task',self.task)
        x = self.backbone(x)
        if self.task == "multi":
            if hasattr(self, 'fpn'):
                x,out_segment= self.fpn(x)
            if hasattr(self, 'head'):
                #print('x',x.shape)
                x = self.head(x)
        #print('out_segment',out_segment.shape)
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
                torch.cuda.synchronize()
                time1 = time.time()
                preds,preds_segment = self(meta['img'])
                torch.cuda.synchronize()
                time2 = time.time()
                print('forward1 time: {:.3f}s'.format((time2 - time1)), end=' | ')
                results = self.head.post_process(preds, meta)
                torch.cuda.synchronize()
                print('decode time: {:.3f}s'.format((time.time() - time2)), end=' | ')
                return results,preds_segment
            elif self.task == "segment":
                torch.cuda.synchronize()
                time1 = time.time()
                preds_segment = self(meta['img'])
                torch.cuda.synchronize()
                time2 = time.time()
                #print('forward1 time: {:.3f}s'.format((time2 - time1)), end=' | ')
                #results = self.head.post_process(preds, meta)
                torch.cuda.synchronize()
                #print('decode time: {:.3f}s'.format((time.time() - time2)), end=' | ')
                return preds_segment
            else:
                torch.cuda.synchronize()
                time1 = time.time()
                preds = self(meta['img'])
                torch.cuda.synchronize()
                time2 = time.time()
                print('forward time: {:.3f}s'.format((time2 - time1)), end=' | ')
                results = self.head.post_process(preds, meta)
                torch.cuda.synchronize()
                print('decode time: {:.3f}s'.format((time.time() - time2)), end=' | ')
                return results


    def forward_train(self, gt_meta):
        preds,preds_segment = self(gt_meta['img'])
        #preds_segment=self.segment(gt_meta['img_segment'].float())
        loss, loss_states =self.head.loss(preds,preds_segment, gt_meta)
        return preds, loss, loss_states,preds_segment,gt_meta
