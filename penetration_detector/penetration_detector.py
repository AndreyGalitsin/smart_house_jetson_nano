import RPi.GPIO as GPIO
import time

PIN_NUMBER = 15

GPIO.cleanup()
GPIO.setmode(GPIO.BOARD)
GPIO.setup(PIN_NUMBER, GPIO.IN)
while 1:
    a=time.time()
    x = GPIO.input(PIN_NUMBER)
    #print(time.time()-a)
    if int(x) == 1:
        print('open')
    else:
        print('close')
    time.sleep(2)#print(x)


