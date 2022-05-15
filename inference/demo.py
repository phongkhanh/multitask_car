import cv2
import os
import time
import torch
import argparse
from nanodet.util import cfg, load_config, Logger
from nanodet.model.arch import build_model
from nanodet.util import load_model_weight
from nanodet.data.transform import Pipeline
import time
image_ext = ['.jpg', '.jpeg', '.webp', '.bmp', '.png']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('demo', default='image', help='demo type, eg. image, video and webcam')
    parser.add_argument('--config', help='model config file path')
    parser.add_argument('--model', help='model file path')
    parser.add_argument('--path', default='./demo', help='path to images or video')
    parser.add_argument('--camid', type=int, default=0, help='webcam demo camera id')
    parser.add_argument('--save_result', action='store_true', help='whether to save the inference result of image/video')
    args = parser.parse_args()
    return args
def reverse_one_hot(image):
  image = image.permute(1, 2, 0)
  x = torch.argmax(image, dim=-1)
  return x
def mask_with_color(img, mask, color=(255,255,255)):
    color_mask = np.zeros(img.shape, img.dtype)
    color_mask[:,:] = color
    color_mask = cv2.bitwise_and(color_mask, color_mask, mask=mask)
    return cv2.addWeighted(color_mask, 1, img, 1, 0)
class Predictor(object):
    def __init__(self, cfg, model_path, logger, device='cuda'):
        self.cfg = cfg
        print(cfg)
        self.task=cfg.model.arch.fpn.task
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = build_model(cfg.model)
        ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
        load_model_weight(model, ckpt, logger)
        if cfg.model.arch.backbone.name == 'RepVGG':
            deploy_config = cfg.model
            deploy_config.arch.backbone.update({'deploy': True})
            deploy_model = build_model(deploy_config)
            from nanodet.model.backbone.repvgg import repvgg_det_model_convert
            model = repvgg_det_model_convert(model, deploy_model)
        self.model = model.to(self.device).eval()
        self.pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)
        
    def inference(self, img):
        img_info = {'id': 0}
        if isinstance(img, str):
            img_info['file_name'] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info['file_name'] = None

        height, width = img.shape[:2]
        img_info['height'] = height
        img_info['width'] = width
        meta = dict(img_info=img_info,
                    raw_img=img,
                    img=img)
        meta = self.pipeline(meta, self.cfg.data.val.input_size)
        meta['img'] = torch.from_numpy(meta['img'].transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.task =="multi":
                results,out_segment = self.model.inference(meta)
                val_output =out_segment[0]
                val_output = reverse_one_hot(val_output)
                val_output = np.array(val_output.cpu())
                road_mask=val_output*255
                
                return meta, results,road_mask
            elif self.task =="segment":
                out_segment = self.model.inference(meta)
                val_output =out_segment[0]
                val_output = reverse_one_hot(val_output)
                val_output = np.array(val_output.cpu())
                road_mask=val_output*255
            
                return meta,road_mask
            else:
                results = self.model.inference(meta)
                return meta,results
    def visualize(self, dets, meta, class_names, score_thres, wait=0):
        result_img = self.model.head.show_result(meta['raw_img'], dets, class_names, score_thres=score_thres, show=False)
        return result_img
def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in image_ext:
                image_names.append(apath)
    return image_names
def main():
    args = parse_args()
    local_rank = 0
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    load_config(cfg, args.config)
    task=cfg.model.arch.fpn.task
    logger = Logger(local_rank, use_tensorboard=False)
    predictor = Predictor(cfg, args.model, logger, device='cuda:1')
    logger.log('Press "Esc", "q" or "Q" to exit.')
    current_time = time.localtime()
    if args.demo == 'image':
        if os.path.isdir(args.path):
            files = get_image_list(args.path)
        else:
            files = [args.path]
        files.sort()
        for image_name in files:
            if task == "multi":
                meta, res ,road_mask= predictor.inference(image_name)
                result_image = predictor.visualize(res[0], meta, cfg.class_names, 0.35)
                if args.save_result:
                    result_image = predictor.visualize(res[0], meta, cfg.class_names, 0.35)
                    road_mask = np.asarray(road_mask).astype('uint8')
                    result_frame = mask_with_color(result_image, road_mask, color=(0,255,0))
                    cv2.imwrite("pred_seg1.png", result_frame)
                ch = cv2.waitKey(0)
                if ch == 27 or ch == ord('q') or ch == ord('Q'):
                    break
            elif task == "segment":
                meta, road_mask= predictor.inference(image_name)
                img=cv2.imread(image_name)
                if args.save_result:
                    road_mask = np.asarray(road_mask).astype('uint8')
                    result_frame = mask_with_color(img, road_mask, color=(0,255,0))
                    cv2.imwrite("out_seg.png",result_frame)        
                ch = cv2.waitKey(0)
                if ch == 27 or ch == ord('q') or ch == ord('Q'):
                    break
            else:
                meta, res = predictor.inference(image_name)
                result_frame = predictor.visualize(res[0], meta, cfg.class_names, 0.35)
                if args.save_result:
                    cv2.imwrite("out_detection.png",result_frame)
    elif args.demo == 'video' or args.demo == 'webcam':
        cap = cv2.VideoCapture(args.path if args.demo == 'video' else args.camid)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
        fps = cap.get(cv2.CAP_PROP_FPS)
        vid_writer = cv2.VideoWriter("out_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(width), int(height)))
        fps=[]
        while True:
            ret_val, frame = cap.read()
            if ret_val:
                if task == "multi":
                    start = time.time()
                
                    meta, res,road_mask = predictor.inference(frame)
                    end = time.time()
                    
                    result_frame = predictor.visualize(res[0], meta, cfg.class_names, 0.35)
                    road_mask = np.asarray(road_mask).astype('uint8')
                    result_frame = mask_with_color(result_frame, road_mask, color=(0,255,0))   
                    fps.append(1/(end - start))
                    if args.save_result:
                        vid_writer.write(result_frame)
                        
                    ch = cv2.waitKey(1)
                    if ch == 27 or ch == ord('q') or ch == ord('Q'):
                        break
                elif task == "segment":
                    start = time.time()
                    meta,road_mask = predictor.inference(frame)
                    road_mask = np.asarray(road_mask).astype('uint8')
                    result_frame = mask_with_color(frame, road_mask, color=(0,255,0))
                    end = time.time()
                    fps.append(1/(end - start))
                    if args.save_result:
                        vid_writer.write(result_frame)
                        
                    ch = cv2.waitKey(1)
                    if ch == 27 or ch == ord('q') or ch == ord('Q'):
                        break
                else:
                    start = time.time()
                    meta, res = predictor.inference(frame)
                    result_frame = predictor.visualize(res[0], meta, cfg.class_names, 0.35)
                    end = time.time()
                    fps.append(1/(end - start))
                    
                    if args.save_result:
                        vid_writer.write(result_frame)
                    ch = cv2.waitKey(1)
                    if ch == 27 or ch == ord('q') or ch == ord('Q'):
                        break
            else:
                break

        print('fps mean',np.mean(fps))

if __name__ == '__main__':
    main()
