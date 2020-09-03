import cv2
import time
from resnet import Resnet

if __name__ == '__main__':
    device = "/dev/video0"

    Cam = cv2.VideoCapture(device)
    Cam.set(cv2.CAP_PROP_BUFFERSIZE, 3)
    print(Cam.isOpened())

    counter = 0
    while 1:
        counter += 1
        ret, image = Cam.read()
        resnet = Resnet()

        if counter % 1 == 0:
            counter = 0
            if image is not None:
                resnet.get_weights(image)
                cv2.imshow("lalala", image)
                cv2.waitKey(1)
            else:
                print("cannot receive img from camera")
                Cam = cv2.VideoCapture(device)
                time.sleep(0.01)
        else:
            continue
    