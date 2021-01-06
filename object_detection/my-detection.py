import jetson.inference
import jetson.utils
import datetime
import cv2
import numpy as np
import random

class ObjDet:
    def __init__(self):
        self.net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
        self.classes = ['unlabeled', 'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
        'traffic light','fire hydrant','street sign','stop sign','parking meter','bench','bird','cat','dog','horse',
        'sheep','cow','elephant','bear','zebra','giraffe','hat','backpack','umbrella','shoe','eye glasses',
        'handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove',
        'skateboard','surfboard','tennis racket','bottle','plate','wine glass','cup','fork','knife','spoon',
        'bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake',
        'chair','couch','potted plant','bed','mirror','dining table','window','desk','toilet','door',
        'tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink',
        'refrigerator','blender','book','clock','vase','scissors','teddy bear','hair drier','toothbrush']
    
    def choose_color(self):
        color_1 = (0, 255, 255)
        color_2 = (255, 0, 255)
        color_3 = (255, 255, 0)
        color_4 = (0, 0, 255)
        color_5 = (0, 255, 0)
        color_6 = (255, 0, 0)
        colors = [color_1,color_2,color_3,color_4,color_5,color_6]
        i = random.randint(0,5)
        return colors[i]

    def main(self, frame):
        frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        width = frame.shape[1]
        height = frame.shape[0]
        img = jetson.utils.cudaFromNumpy(frame_rgba)
        detections = self.net.Detect(img, width, height)
        for i in range(len(detections)):
            class_id = int(detections[i].ClassID)
            confidence = str(round(detections[i].Confidence, 2))
            left = detections[i].Left
            right = detections[i].Right
            top =detections[i].Top
            bot = detections[i].Bottom
            color = self.choose_color()
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bot)), color, 3)
            cv2.putText(frame, self.classes[class_id] + ' ' + str(confidence), (int(left), int(top) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            color, 1)

            #print(class_id, confidence, left, right, top, bot)
        
        return frame



if __name__ == "__main__":
    object_det = ObjDet()
    cam = cv2.VideoCapture(0)
    counter = 0
    while True:
        counter += 1
        _, frame = cam.read()

        if counter % 30 == 0:
            counter = 0
            if frame is not None:
                last_frame = object_det.main(frame)

                cv2.imshow('Object detection', last_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'): 	
                    break
            else:
                print("cannot receive img from camera")
                cam = cv2.VideoCapture(0)
                time.sleep(0.01)
        else:
            continue
    cv2.destroyAllWindows()