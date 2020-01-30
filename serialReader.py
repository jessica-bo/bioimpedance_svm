"""
Created on Fri Jan 29 2020

"""

# IMPORTED LIBRARIES

import os
import datetime, time
import sys

import pandas
import numpy as np
import matplotlib.pyplot as plt

import serial
from serial.tools import list_ports

# MAIN PROGRAM

if __name__ == "__main__":

    listed_ports = list_ports.comports()

    print("COM Ports Connected: ")

    for port in listed_ports:
        print("{}".format(port))

    port_selected = input("Selected port: ")
    material = input("Target material (blood, fat, muscle): ")

    if port_selected != "":
        ser = serial.Serial(port = "COM{}".format(port_selected), baudrate = 9600, bytesize = 8)

        # this will store the line
        dataStream = []

        byteLine = []

        for i in range(100):
            for c in ser.read():
                byteLine.append(c)
                if c == '\n':
                    dataStream.append(byteLine)
                    print(byteLine)
                    byteLine = []

        np.savetxt("{}.csv".format(material), np.asarray(dataStream), delimiter=",")

        ser.close()


