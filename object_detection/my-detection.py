import jetson.inference
import jetson.utils
import datetime
import cv2

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
camera = jetson.utils.gstCamera(320, 240, "/dev/video0")
#camera = jetson.utils.gstCamera(1280, 720, "/dev/video0")
display = jetson.utils.glDisplay()

while display.IsOpen():
	img, width, height = camera.CaptureRGBA(zeroCopy=1)
	#img, width, height = jetson.utils.loadImageRGBA('./test.jpeg')
	t1=datetime.datetime.now()
	detections = net.Detect(img, width, height)
	img_numpy = jetson.utils.cudaToNumpy(img, width, height, 4)
	img_numpy = cv2.cvtColor(img_numpy, cv2.COLOR_BGR2RGB)
	t2=datetime.datetime.now()
	#print('time for detection', t2-t1)

	#cv2.imwrite('./photo.jpg', img_numpy)
	display.RenderOnce(img, width, height)
	#display.SetTitle("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))



