#!/usr/bin/env python
# license removed for brevity
import rospy
import numpy as np
#from std_msgs.msg import String
from rospy_tutorials.msg import Floats
import cv2 as cv

def send_data():
    pub = rospy.Publisher('send_image_data', Floats, queue_size=10)
    rospy.init_node('image_data_talker', anonymous=True)
    rate = rospy.Rate(100) # 1Hz
    image_path = "/home/mendez/catkin_build_ws/src/numpy_tutorial/data/pizza_samples/sample_01.jpg"
    img = cv.imread(image_path, cv.IMREAD_COLOR)
    resized = cv.resize(img, (224, 224), interpolation = cv.INTER_AREA)
    a = np.array(resized)
    a = a / 255.0
    a = a.flatten()

    while not rospy.is_shutdown():
        print("Publishing data....")
        pub.publish(a)
        print("Finishing publishing data....")
        rate.sleep()


if __name__ == '__main__':
    try:
        send_data()
    except rospy.ROSInterruptException:
        pass
