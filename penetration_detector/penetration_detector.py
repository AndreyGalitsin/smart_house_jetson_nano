import RPi.GPIO as GPIO


PIN_NUMBER = 15

GPIO.cleanup()
GPIO.setmode(GPIO.BOARD)
GPIO.setup(PIN_NUMBER, GPIO.IN)
while 1:
    x = GPIO.input(PIN_NUMBER)
    print(x)


