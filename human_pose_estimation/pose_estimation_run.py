import json
import trt_pose.coco
import trt_pose.models
import torch
import torch2trt
import numpy as np

with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose)

WIDTH = 224
HEIGHT = 224

data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()


OPTIMIZED_MODEL = './models/resnet18_baseline_att_224x224_A_epoch_249_trt.pth'



from torch2trt import TRTModule

model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))
'''
import time

t0 = time.time()
torch.cuda.current_stream().synchronize()
for i in range(50):
    y = model_trt(data)
torch.cuda.current_stream().synchronize()
t1 = time.time()

print(50.0 / (t1 - t0))
'''
import cv2
import torchvision.transforms as transforms
import PIL.Image

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')

def preprocess(image):
    global device
    device = torch.device('cuda')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects

parse_objects = ParseObjects(topology)
draw_objects = DrawObjects(topology)



def func_1():
    cv2.namedWindow('Detecto')
    device = 0
    try:
        video = cv2.VideoCapture(device)
        video.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        print(video.isOpened())
    except:
        print('No webcam available.')
        return
    counter = 0
    while True:
        counter += 1
        ret, frame = video.read()
        if not ret:
            continue

        if counter % 1 == 0:
            counter = 0

            data = preprocess(frame)
            cmap, paf = model_trt(data)
            cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
            counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
            draw_objects(frame, counts, objects, peaks)
            cv2.imwrite('./picture_2.jpeg', frame)
            print('frame', frame)
            print('counts', counts)
            print('objects', objects)
            print('peaks', peaks)

            cv2.imshow('Detecto', frame)

            # If the 'q' or ESC key is pressed, break from the loop
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
        else:
            continue
        
    cv2.destroyWindow('Detecto')
    video.release()

def func_2():
    from jetcam.usb_camera import USBCamera
    from jetcam.utils import bgr8_to_jpeg

    camera = USBCamera(capture_device = 0)

    camera.running = True

    import ipywidgets
    from IPython.display import display

    image_w = ipywidgets.Image(format='jpeg')

    display(image_w)


    def execute(change):
        image = change['new']
        data = preprocess(image)
        cmap, paf = model_trt(data)
        cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
        counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
        draw_objects(image, counts, objects, peaks)
        cv2.imwrite('./picture.jpeg', image)
        print('frame', image)
        print('counts', counts)
        print('objects', objects)
        print('peaks', peaks)

        cv2.imshow('Detecto', image)

        # If the 'q' or ESC key is pressed, break from the loop
        #key = cv2.waitKey(1) & 0xFF
        #if key == ord('q') or key == 27:
        #    break

        #image_w.value = bgr8_to_jpeg(image[:, ::-1, :])
        #decoded = cv2.imdecode(np.frombuffer(image_w.value, np.uint8), cv2.IMREAD_COLOR)

    camera.observe(execute, names='value')

func_2()