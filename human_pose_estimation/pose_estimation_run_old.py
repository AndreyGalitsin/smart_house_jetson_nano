import json
import trt_pose.coco
import trt_pose.models
import torch
import torch2trt
from torch2trt import TRTModule
import time
import cv2
import torchvision.transforms as transforms
import PIL.Image, PIL.ImageDraw
from trt_pose.parse_objects import ParseObjects
import argparse
import os.path

class PoseEstimation:
    def __init__(self):
        with open('human_pose.json', 'r') as f:
            human_pose = json.load(f)

        topology = trt_pose.coco.coco_category_to_topology(human_pose)
        print('------ model = resnet--------')
        OPTIMIZED_MODEL = './models/resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
        self.WIDTH = 224
        self.HEIGHT = 224

        print('Loading model')
        self.model_trt = TRTModule()
        self.model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))
        print('model was loaded')

        self.mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
        self.std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
        self.device = torch.device('cuda')

        self.parse_objects = ParseObjects(topology)

    '''
    img is PIL format
    '''
    def draw_keypoints(self, img, key):
        thickness = 5
        w, h = img.size
        draw = PIL.ImageDraw.Draw(img)
        #draw Rankle -> RKnee (16-> 14)
        if all(key[16]) and all(key[14]):
            draw.line([ int(key[16][2] * w), int(key[16][1] * h), int(key[14][2] * w), int(key[14][1] * h)],width = thickness, fill=(51,51,204))
        #draw RKnee -> Rhip (14-> 12)
        if all(key[14]) and all(key[12]):
            draw.line([ int(key[14][2] * w), int(key[14][1] * h), int(key[12][2] * w), int(key[12][1] * h)],width = thickness, fill=(51,51,204))
        #draw Rhip -> Lhip (12-> 11)
        if all(key[12]) and all(key[11]):
            draw.line([ int(key[12][2] * w), int(key[12][1] * h), int(key[11][2] * w), int(key[11][1] * h)],width = thickness, fill=(51,51,204))
        #draw Lhip -> Lknee (11-> 13)
        if all(key[11]) and all(key[13]):
            draw.line([ int(key[11][2] * w), int(key[11][1] * h), int(key[13][2] * w), int(key[13][1] * h)],width = thickness, fill=(51,51,204))
        #draw Lknee -> Lankle (13-> 15)
        if all(key[13]) and all(key[15]):
            draw.line([ int(key[13][2] * w), int(key[13][1] * h), int(key[15][2] * w), int(key[15][1] * h)],width = thickness, fill=(51,51,204))

        #draw Rwrist -> Relbow (10-> 8)
        if all(key[10]) and all(key[8]):
            draw.line([ int(key[10][2] * w), int(key[10][1] * h), int(key[8][2] * w), int(key[8][1] * h)],width = thickness, fill=(255,255,51))
        #draw Relbow -> Rshoulder (8-> 6)
        if all(key[8]) and all(key[6]):
            draw.line([ int(key[8][2] * w), int(key[8][1] * h), int(key[6][2] * w), int(key[6][1] * h)],width = thickness, fill=(255,255,51))
        #draw Rshoulder -> Lshoulder (6-> 5)
        if all(key[6]) and all(key[5]):
            draw.line([ int(key[6][2] * w), int(key[6][1] * h), int(key[5][2] * w), int(key[5][1] * h)],width = thickness, fill=(255,255,0))
        #draw Lshoulder -> Lelbow (5-> 7)
        if all(key[5]) and all(key[7]):
            draw.line([ int(key[5][2] * w), int(key[5][1] * h), int(key[7][2] * w), int(key[7][1] * h)],width = thickness, fill=(51,255,51))
        #draw Lelbow -> Lwrist (7-> 9)
        if all(key[7]) and all(key[9]):
            draw.line([ int(key[7][2] * w), int(key[7][1] * h), int(key[9][2] * w), int(key[9][1] * h)],width = thickness, fill=(51,255,51))

        #draw Rshoulder -> RHip (6-> 12)
        if all(key[6]) and all(key[12]):
            draw.line([ int(key[6][2] * w), int(key[6][1] * h), int(key[12][2] * w), int(key[12][1] * h)],width = thickness, fill=(153,0,51))
        #draw Lshoulder -> LHip (5-> 11)
        if all(key[5]) and all(key[11]):
            draw.line([ int(key[5][2] * w), int(key[5][1] * h), int(key[11][2] * w), int(key[11][1] * h)],width = thickness, fill=(153,0,51))


        #draw nose -> Reye (0-> 2)
        if all(key[0][1:]) and all(key[2]):
            draw.line([ int(key[0][2] * w), int(key[0][1] * h), int(key[2][2] * w), int(key[2][1] * h)],width = thickness, fill=(219,0,219))

        #draw Reye -> Rear (2-> 4)
        if all(key[2]) and all(key[4]):
            draw.line([ int(key[2][2] * w), int(key[2][1] * h), int(key[4][2] * w), int(key[4][1] * h)],width = thickness, fill=(219,0,219))

        #draw nose -> Leye (0-> 1)
        if all(key[0][1:]) and all(key[1]):
            draw.line([ int(key[0][2] * w), int(key[0][1] * h), int(key[1][2] * w), int(key[1][1] * h)],width = thickness, fill=(219,0,219))

        #draw Leye -> Lear (1-> 3)
        if all(key[1]) and all(key[3]):
            draw.line([ int(key[1][2] * w), int(key[1][1] * h), int(key[3][2] * w), int(key[3][1] * h)],width = thickness, fill=(219,0,219))

        #draw nose -> neck (0-> 17)
        if all(key[0][1:]) and all(key[17]):
            draw.line([ int(key[0][2] * w), int(key[0][1] * h), int(key[17][2] * w), int(key[17][1] * h)],width = thickness, fill=(255,255,0))
        return img

    '''
    hnum: 0 based human index
    kpoint : index + keypoints (float type range : 0.0 ~ 1.0 ==> later multiply by image width, height)
    '''
    def get_keypoint(self, humans, hnum, peaks):
        #check invalid human index
        kpoint = []
        human = humans[0][hnum]
        C = human.shape[0]
        for j in range(C):
            k = int(human[j])
            if k >= 0:
                peak = peaks[0][j][k]   # peak[1]:width, peak[0]:height
                peak = (j, float(peak[0]), float(peak[1]))
                kpoint.append(peak)
                print('index:%d : success [%5.3f, %5.3f]'%(j, peak[1], peak[2]) )
            else:    
                peak = (j, None, None)
                kpoint.append(peak)
                print('index:%d : None'%(j) )
        return kpoint

    def preprocess(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = PIL.Image.fromarray(image)
        image = transforms.functional.to_tensor(image).to(self.device)
        image.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])
        return image[None, ...]

    '''
    Draw to original image
    '''
    def execute(self, img, org):
        start = time.time()
        data = self.preprocess(img)
        cmap, paf = self.model_trt(data)
        cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
        end = time.time()
        counts, objects, peaks = self.parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
        for i in range(counts[0]):
            print("Human index:%d "%( i ))
            kpoint = self.get_keypoint(objects, i, peaks)
            #print(kpoint)
            org = self.draw_keypoints(org, kpoint)
        print("Human count:%d len:%d "%(counts[0], len(counts)))
        print('===== Net FPS :%f ====='%( 1 / (end - start)))
        return org

    def main(self, image):
        pilimg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pilimg = PIL.Image.fromarray(pilimg)
        orgimg = pilimg.copy()

        image = cv2.resize(src, dsize=(self.WIDTH, self.HEIGHT), interpolation=cv2.INTER_AREA)
        res = self.execute(image, orgimg)
        return res



if __name__ == "__main__":
    pose_estimation = PoseEstimation()

    parser = argparse.ArgumentParser(description='TensorRT pose estimation run')
    parser.add_argument('--image', type=str, default='./test.jpg')
    args = parser.parse_args()


    src = cv2.imread(args.image, cv2.IMREAD_COLOR)
    

    #img = image.copy()
    pilimg = pose_estimation.main(src)

    pilimg.save('./%s.png'%('aaa'))
