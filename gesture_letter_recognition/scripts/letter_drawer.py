#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage

drawing = False
last_point = None
canvas = np.zeros((256, 256, 3), dtype=np.uint8)

def mouse_callback(event, x, y, flags, param):
    global drawing, last_point, canvas
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        last_point = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.line(canvas, last_point, (x, y), (255, 255, 255), 4)
        last_point = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        last_point = None
        # Publish the image
        publish_image(canvas)

def publish_image(img):
    msg = CompressedImage()
    msg.header.stamp = rospy.Time.now()
    msg.format = "jpeg"
    msg.data = np.array(cv2.imencode('.jpg', img)[1]).tobytes()
    pub.publish(msg)
    print("Image published!")

if __name__ == "__main__":
    rospy.init_node("letter_drawer")
    pub = rospy.Publisher("drawn_letter/image/compressed", CompressedImage, queue_size=1)
    cv2.namedWindow("Draw Letter")
    cv2.setMouseCallback("Draw Letter", mouse_callback)

    while not rospy.is_shutdown():
        cv2.imshow("Draw Letter", canvas)
        key = cv2.waitKey(10) & 0xFF
        if key == ord('c'):
            canvas[:] = 0
        elif key == 27:  # ESC to exit
            break

    cv2.destroyAllWindows()