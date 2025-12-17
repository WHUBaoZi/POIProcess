#!/home/unitree/poi_detect/poi_detect/venv/bin/python

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import rospy
from std_msgs.msg import String

import numpy as np
import poi_detect



def image_callback(msg):
    np_arr = np.frombuffer(msg.data, np.unit8)




if __name__ == "__main__":
    result = poi_detect.poi_detect("/home/unitree/poi_detect/poi_detect/src/poi_algorithm/testdata")
    print("Test result: %s" % str(result))