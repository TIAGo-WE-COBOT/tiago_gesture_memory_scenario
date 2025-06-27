#!/usr/bin/env python3

import os
import numpy as np
import cv2
import torch

import rospy
import rospkg
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
from my_ros_utils.image_conversion import decode_CompressedImage_RGB

from cnn import CNNClassifier

class LetterClassifier:
    def __init__(self):
        # Load model weights from file
        r = rospkg.RosPack()
        pkg_root = r.get_path('gesture_letter_recognition')
        model_path = os.path.join(pkg_root, 'models', 'pytorch_weights.pth')
        # Initialize the CNN model
        self.model = CNNClassifier()
        self.model.load_state_dict(torch.load(model_path, 
                                              map_location=torch.device('cpu')))
        self.model.eval()

        # Subscribe to the binary image topic
        self.img_sub = rospy.Subscriber('binary_img', CompressedImage, self.classify_image)
        # Publisher to send the classified letter
        self.letter_pub = rospy.Publisher('letter', String, queue_size=1)

    def classify_image(self, cmpr_img_msg):
        img = decode_CompressedImage_RGB(cmpr_img_msg) # img is a np.ndarray
        if img is None or np.sum(img) == 0:
            rospy.logwarn("Received empty or invalid image")
            return
        # Ensure the image is binary (0s and 255s)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Preprocess the image
        rows, cols = np.nonzero(img)
        margin = 40 # TODO. Make it a parameter
        img = img[np.min(rows) - margin: np.max(rows) + margin,
                  np.min(cols) - margin: np.max(cols) + margin]
        img = img.astype('float32') / 255.0
        print(img.shape)
        img = cv2.resize(img, (20, 20))  # Back to 28x28 as per training
        img = torch.tensor(img).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        print(img.shape)  # Debugging line to check the shape of the image tensor
        with torch.no_grad():
            output = self.model(img)
            y_pred = output.argmax(dim=1).item()
            confidence = output.max().item() * 100

            letter = chr(y_pred + 65)  # Convert to letter (A=0, B=1, ..., Z=25)
            rospy.loginfo(f'Predicted letter: {letter} with confidence {confidence:.2f}%')

            # Publish the result
            self.letter_pub.publish(letter)

if __name__ == "__main__":
    rospy.init_node('letter_classifier_node', anonymous=True)
    classifier = LetterClassifier()
    rospy.loginfo("Letter classifier node started")
    rospy.spin()