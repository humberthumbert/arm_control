import pyfirmata
import time
import serial
import numpy as np
from RobotArm import *
# board = pyfirmata.Arduino('/dev/cu.usbserial-1432230')
ser = serial.Serial('/dev/cu.usbserial-1432420', 9600)
c = ""
arm1 = RobotArm(0, 12, 10, 8)
result = "1:150;150;150;150;150;150;2:150;150;150;150;150;170;3:150;150;150;150;150;150;:"
ser.write(str.encode(result))
val = "a"
while (ser.readline(30) == b'Poses Complete\r\n') and (val == "a"):
    ser.write(str.encode(result))
    print("Write Again")
    val = input("--> ")

# while True:

#     c = input("Val: ")
#     if c == 'q':
#         break
#     elif len(c.split(',')) == 3:
#         x, y, alpha = c.split(',')
#         s = arm1.getReverseString(float(x), float(y), float(alpha))
#         if s != str.encode('F'):
#             print(s)
#             arm1.draw(x,y)
#          #11   ser.write(s)
#         else:
#             print("Failed: theta1: {}, theta2: {}, theta3: {}".format(arm1.theta1, arm1.theta2, arm1.theta3))
#         time.sleep(1)

