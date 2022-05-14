# Copyright 2021 RangiLyu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import os
import warnings
import json
import torch
import logging

from pytorch_lightning import LightningModule
from typing import Any, List
from nanodet.util import mkdir, gather_results
from ..model.arch.one_stage_detector import OneStageDetector
from ..model.arch import build_model
import matplotlib.pyplot as plt
import numpy as np
import cv2
def reverse_one_hot(image):
        # Convert output of model to predicted class 
  image = image.permute(1, 2, 0)
  x = torch.argmax(image, dim=-1)
  return x
def compute_accuracy(pred, label):

  pred = pred.flatten()
  label = label.flatten()
  total = len(label)
  count = 0.0
  for i in range(total):
    if pred[i] == label[i]:
      count = count + 1.0
                #print(count)
  return float(count) / float(total)
def fast_hist(a, b, n):
  k = (a >= 0) & (a < n)
  return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)
    
def per_class_iu(hist):
  epsilon = 1e-5
  return (np.diag(hist) + epsilon) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + epsilon)
class TrainingTask(LightningModule):
    """
    Pytorch Lightning module of a general training task.
    Including training, evaluating and testing.
    Args:
        cfg: Training configurations
        evaluator: Evaluator for evaluating the model performance.
    """

    def __init__(self, cfg, evaluator=None):
        super(TrainingTask, self).__init__()
        self.cfg = cfg
        self.model = build_model(cfg.model)
        self.evaluator = evaluator
        self.save_flag = -10
        self.log_style = 'NanoDet'  # Log style. Choose between 'NanoDet' or 'Lightning'
        # TODO: use callback to log
        self.val_losses = []
        self.train_losses =[]
        self.epoch_loss_train= []
        self.epoch_loss_val =[]
        self.acc=[]
    def forward(self, x):
        x = self.model(x)
        return x

    @torch.no_grad()
    def predict(self, batch, batch_idx=None, dataloader_idx=None):
        preds = self.forward(batch['img'])
        results = self.model.head.post_process(preds, batch)
        return results

    def on_train_start(self) -> None:
        self.lr_scheduler.last_epoch = self.current_epoch-1

    def training_step(self, batch, batch_idx):
        preds, loss, loss_states, preds_segment,gt_meta= self.model.forward_train(batch)
        # log train losses
        if self.log_style == 'Lightning':
            self.log('lr', self.optimizers().param_groups[0]['lr'], on_step=True, on_epoch=False, prog_bar=True)
            for k, v in loss_states.items():
                self.log('Train/'+k, v, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        elif self.log_style == 'NanoDet' and self.global_step % self.cfg.log.interval == 0:
            lr = self.optimizers().param_groups[0]['lr']
            log_msg = 'Train|Epoch{}/{}|Iter{}({})| lr:{:.2e}| loss_train:{:.3f} |'.format(self.current_epoch+1,
                self.cfg.schedule.total_epochs, self.global_step, batch_idx, lr,loss)
            self.scalar_summary('Train_loss/lr', 'Train', lr, self.global_step)
            for l in loss_states:
                log_msg += '{}:{:.4f}| '.format(l, loss_states[l].mean().item())
                self.scalar_summary('Train_loss/' + l, 'Train', loss_states[l].mean().item(), self.global_step)
            self.info(log_msg)
        self.epoch_loss_train.append(loss.item())
        #loss_train_mean = np.mean(epoch_loss)
        #self.train_losses.append(loss_train_mean.item())
        return loss
    
        
    def training_epoch_end(self, outputs: List[Any]) -> None:
        #print('current_epoch',self.current_epoch+1)
        #print('val_intervals',self.cfg.schedule.val_intervals)
        if ((self.current_epoch+1) % (self.cfg.schedule.val_intervals) == 0):
          #print('end 1 epoch')
          loss_train_mean = 0
          loss_train_mean = np.mean( self.epoch_loss_train)
          self.train_losses.append(loss_train_mean.item())
          self.epoch_loss_train=[]
        if True:
          plt.figure(figsize=(10,5))
          plt.title("Training and Validation Loss")
          plt.plot(self.val_losses,label="val")
          plt.plot(self.train_losses,label="train")
          plt.xlabel("iterations")
          plt.ylabel("Loss")
          plt.legend()
          plt.savefig(os.path.join(self.cfg.save_dir, 'loss.png'))
        if self.current_epoch+1 >= self.cfg.schedule.total_epochs:
          print(self.val_losses)
          print(self.train_losses)
          print('num of epoch of val',len(self.val_losses)*self.cfg.schedule.val_intervals)
          print('num of epoch of train',len(self.train_losses)*self.cfg.schedule.val_intervals)
          plt.figure(figsize=(10,5))
          plt.title("Training and Validation Loss")
          plt.plot(self.val_losses,label="val")
          plt.plot(self.train_losses,label="train")
          plt.xlabel("iterations")
          plt.ylabel("Loss")
          plt.legend()
          plt.savefig(os.path.join(self.cfg.save_dir, 'loss.png'))
        self.trainer.save_checkpoint(os.path.join(self.cfg.save_dir, 'model_last.ckpt'))
        self.lr_scheduler.step()

    def validation_step(self, batch, batch_idx):
        preds, loss, loss_states,preds_segment,gt_meta = self.model.forward_train(batch)
        val_output =preds_segment[0]
        #val_output =torch.squeeze(preds_segment)
        val_output = reverse_one_hot(val_output)
        val_output = np.array(val_output.cpu())
        cv2.imwrite("val.png",val_output*255)
        val_label=gt_meta['masks']
        val_label=val_label[0]
        #val_label = val_label.squeeze()
        val_label = np.array(val_label.cpu())
        accuracy = compute_accuracy(val_output, val_label)
        #hist += fast_hist(val_label.flatten(), val_output.flatten(), 2)
        print('accuracy',accuracy)
        #print('validation_step')
        if self.log_style == 'Lightning':
            self.log('Val/loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=False)
            for k, v in loss_states.items():
                self.log('Val/' + k, v, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        elif self.log_style == 'NanoDet' and batch_idx % self.cfg.log.interval == 0:
            lr = self.optimizers().param_groups[0]['lr']
            log_msg = 'Val|Epoch{}/{}|Iter{}({})| lr:{:.2e}|loss_val:{:.3f}| '.format(self.current_epoch+1,
                self.cfg.schedule.total_epochs, self.global_step, batch_idx, lr,loss)
            for l in loss_states:
                log_msg += '{}:{:.4f}| '.format(l, loss_states[l].mean().item())
            self.info(log_msg)

        dets = self.model.head.post_process(preds, batch)
        self.epoch_loss_val.append(loss.item())
        self.acc.append(accuracy)
        print("mean ",np.mean(self.acc))
        return dets

    def validation_epoch_end(self, validation_step_outputs):
        """
        Called at the end of the validation epoch with the outputs of all validation steps.
        Evaluating results and save best model.
        Args:
            validation_step_outputs: A list of val outputs

        """
        loss_val_mean = np.mean( self.epoch_loss_val)
        self.val_losses.append(loss_val_mean.item())
        self.epoch_loss_val=[]
        results = {}
        #print('validation_epoch_end')
        
        for res in validation_step_outputs:
            results.update(res)
        all_results = gather_results(results)
        if all_results:
            eval_results = self.evaluator.evaluate(all_results, self.cfg.save_dir, rank=self.local_rank)
            metric = eval_results[self.cfg.evaluator.save_key]
            best_save_path = os.path.join(self.cfg.save_dir, 'model_best')
            save_name = '{}.pth.tar'.format(self.current_epoch+1)
            self.trainer.save_checkpoint(os.path.join(best_save_path,save_name))
            # save best model
            if metric > self.save_flag:
                #print("save")
                self.save_flag = metric
                best_save_path = os.path.join(self.cfg.save_dir, 'model_best')
                mkdir(self.local_rank, best_save_path)
                #save_name = '{}.pth.tar'.format(self.current_epoch+1)
                self.trainer.save_checkpoint(os.path.join(best_save_path, "model_best.ckpt"))
                txt_path = os.path.join(best_save_path, "eval_results.txt")
                if self.local_rank < 1:
                    with open(txt_path, "a") as f:
                        f.write("Epoch:{}\n".format(self.current_epoch+1))
                        for k, v in eval_results.items():
                            f.write("{}: {}\n".format(k, v))
            else:
                warnings.warn('Warning! Save_key is not in eval results! Only save model last!')
            if self.log_style == 'Lightning':
                for k, v in eval_results.items():
                    self.log('Val_metrics/' + k, v, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            elif self.log_style == 'NanoDet':
                for k, v in eval_results.items():
                    self.scalar_summary('Val_metrics/' + k, 'Val', v, self.current_epoch+1)
        else:
            self.info('Skip val on rank {}'.format(self.local_rank))

    def test_step(self, batch, batch_idx):
        dets = self.predict(batch, batch_idx)
        return dets

    def test_epoch_end(self, test_step_outputs):
        print('tesst_epoch_end')
        results = {}
        for res in test_step_outputs:
            results.update(res)
        all_results = gather_results(results)
        if all_results:
            res_json = self.evaluator.results2json(all_results)
            json_path = os.path.join(self.cfg.save_dir, 'results.json')
            json.dump(res_json, open(json_path, 'w'))

            if self.cfg.test_mode == 'val':
                eval_results = self.evaluator.evaluate(all_results, self.cfg.save_dir, rank=self.local_rank)
                txt_path = os.path.join(self.cfg.save_dir, "eval_results.txt")
                with open(txt_path, "a") as f:
                    for k, v in eval_results.items():
                        f.write("{}: {}\n".format(k, v))
        else:
            self.info('Skip test on rank {}'.format(self.local_rank))

    def configure_optimizers(self):
        """
        Prepare optimizer and learning-rate scheduler
        to use in optimization.

        Returns:
            optimizer
        """
        optimizer_cfg = copy.deepcopy(self.cfg.schedule.optimizer)
        name = optimizer_cfg.pop('name')
        build_optimizer = getattr(torch.optim, name)
        optimizer = build_optimizer(params=self.parameters(), **optimizer_cfg)

        schedule_cfg = copy.deepcopy(self.cfg.schedule.lr_schedule)
        name = schedule_cfg.pop('name')
        build_scheduler = getattr(torch.optim.lr_scheduler, name)
        self.lr_scheduler = build_scheduler(optimizer=optimizer, **schedule_cfg)
        # lr_scheduler = {'scheduler': self.lr_scheduler,
        #                 'interval': 'epoch',
        #                 'frequency': 1}
        # return [optimizer], [lr_scheduler]

        return optimizer

    def optimizer_step(self,
                       epoch=None,
                       batch_idx=None,
                       optimizer=None,
                       optimizer_idx=None,
                       optimizer_closure=None,
                       on_tpu=None,
                       using_native_amp=None,
                       using_lbfgs=None):
        """
        Performs a single optimization step (parameter update).
        Args:
            epoch: Current epoch
            batch_idx: Index of current batch
            optimizer: A PyTorch optimizer
            optimizer_idx: If you used multiple optimizers this indexes into that list.
            optimizer_closure: closure for all optimizers
            on_tpu: true if TPU backward is required
            using_native_amp: True if using native amp
            using_lbfgs: True if the matching optimizer is lbfgs
        """
        # warm up lr
        if self.trainer.global_step <= self.cfg.schedule.warmup.steps:
            if self.cfg.schedule.warmup.name == 'constant':
                warmup_lr = self.cfg.schedule.optimizer.lr * self.cfg.schedule.warmup.ratio
            elif self.cfg.schedule.warmup.name == 'linear':
                k = (1 - self.trainer.global_step / self.cfg.schedule.warmup.steps) * (1 - self.cfg.schedule.warmup.ratio)
                warmup_lr = self.cfg.schedule.optimizer.lr * (1 - k)
            elif self.cfg.schedule.warmup.name == 'exp':
                k = self.cfg.schedule.warmup.ratio ** (1 - self.trainer.global_step / self.cfg.schedule.warmup.steps)
                warmup_lr = self.cfg.schedule.optimizer.lr * k
            else:
                raise Exception('Unsupported warm up type!')
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        items.pop("loss", None)
        return items

    def scalar_summary(self, tag, phase, value, step):
        """
        Write Tensorboard scalar summary log.
        Args:
            tag: Name for the tag
            phase: 'Train' or 'Val'
            value: Value to record
            step: Step value to record

        """
        if self.local_rank < 1:
            self.logger.experiment.add_scalars(tag, {phase: value}, step)

    def info(self, string):
        if self.local_rank < 1:
            logging.info(string)








