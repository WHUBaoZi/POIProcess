import os
import sys
import rospy
from std_msgs.msg import String


import poi_detect

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    result = poi_detect.poi_detect("../../testdata")
    rospy.loginfo("Test result: %s" % str(result))