from serial import Serial
import json

while 1:
    with Serial('/dev/ttyACM0', 9600) as ser:
        line = ser.readline().decode().rstrip()
        try:
            data = json.loads(line)
            print(data)
        except:
            pass
