import json
import trt_pose.coco
import trt_pose.models
import torch
import torch2trt
import numpy as np
from torch2trt import TRTModule
import cv2
import torchvision.transforms as transforms
import PIL.Image
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
from jetcam.usb_camera import USBCamera
from jetcam.utils import bgr8_to_jpeg
import ipywidgets
from IPython.display import display


class PoseEstimation:
    def __init__(self):
        with open('human_pose.json', 'r') as f:
            self.human_pose = json.load(f)

        self.topology = trt_pose.coco.coco_category_to_topology(self.human_pose)
        self.WIDTH = 224
        self.HEIGHT = 224

        OPTIMIZED_MODEL = './models/resnet18_baseline_att_224x224_A_epoch_249_trt.pth'

        self.model_trt = TRTModule()
        self.model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))

        self.mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
        self.std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
        self.device = torch.device('cuda')

        self.parse_objects = ParseObjects(self.topology)
        self.draw_objects = DrawObjects(self.topology)

    def preprocess(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = PIL.Image.fromarray(image)
        image = transforms.functional.to_tensor(image).to(self.device)
        image.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])
        
        return image[None, ...]

    def execute(self, image):
        data = preprocess(image)
        cmap, paf = self.model_trt(data)
        cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
        counts, objects, peaks = self.parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
        self.draw_objects(image, counts, objects, peaks)
        #cv2.imwrite('./picture.jpeg', image)
  
        print('peaks', peaks.shape, peaks[:, :, 0, :], peaks[:, :, 10, :])
        return image

if __name__ == "__main__":
    pose_estimation = PoseEstimation()

    cam = cv2.VideoCapture(0)
    process_this_frame = True
    while True:
        _, frame = cam.read()
        if process_this_frame:
            last_frame = pose_estimation.esecute(frame)
        cv2.imshow('pose estimation', last_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): 	
            break
        process_this_frame = not process_this_frame
    cv2.destroyAllWindows()


'''
def func_2():
    camera = USBCamera(capture_device = 0)
    camera.running = True
    image_w = ipywidgets.Image(format='jpeg')
    display(image_w)

    def execute(change):
        image = change['new']
        data = preprocess(image)
        cmap, paf = self.model_trt(data)
        cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
        counts, objects, peaks = self.parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
        self.draw_objects(image, counts, objects, peaks)
        cv2.imwrite('./picture.jpeg', image)
  
        print('peaks', peaks.shape, peaks[:, :, 0, :], peaks[:, :, 10, :])

    camera.observe(execute, names='value')

func_2()
'''