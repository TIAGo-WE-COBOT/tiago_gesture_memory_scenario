#!/usr/bin/env python3

import os
import cv2
import numpy as np
from keras.models import load_model
from skimage.transform import resize

import rospy
import rospkg
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String

from decoder import decode_Image_RGB

class LetterClassifier:
	def __init__(self):
        # Load the model to classify the letters
        r = rospkg.RosPack()
		pkg_root = r.get_path('gesture_letter_recognition')
		self.model = load_model(os.path.join(pkg_root, 
									         'models', 
											 'letter_classifier.h5'
											 ))
		# Subscribe to the binary image topic
		self.img_sub = rospy.Subscriber('binary_img', 
                                        CompressedImage, 
                                        self.classify_image
                                        )
		# Publisher to send the classified letter
		self.letter_pub = rospy.Publisher('letter', String, queue_size=1)
	
		
	def classify_image(self,img_msg):
		
		byte_array = np.fromstring(img_msg.data, np.uint8)
		img = cv2.imdecode(byte_array, cv2.IMREAD_COLOR)
		
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		
		
		rows,cols = np.nonzero(img)

		margin = 40
		
		if np.sum(img)>0:
		
			img = img[np.min(rows) - margin : np.max(rows) + margin,
          	np.min(cols)- margin : np.max(cols) + margin]

			img = img.astype('float32')/255
		
			image = self.crop_square(img,28)

			image = resize(image, (28, 28,1))
			y_pred = self.model.predict(image.reshape(-1, 28, 28,1))
		
			print('\n\n')
	
			print('LA PREDIZIONE E', chr(y_pred.argmax()+65), 'AL', (y_pred.max()*100),'%')
		
			self.letter_pub.publish(y_pred.argmax())
		
	def crop_square(self,img,size,interpolation=cv2.INTER_AREA):
		h, w = img.shape[:2]
		min_size = np.amin([h,w])
		crop_img = img[int(h/2-min_size/2):int(h/2+min_size/2), int(w/2-min_size/2):int(w/2+min_size/2)]
		resized = cv2.resize(crop_img, (size, size), interpolation=interpolation)

		return resized
		
		
if __name__=='__main__':
	rospy.init_node('image_classifier')
	letter_tracker = LetterClassifier()
	rospy.spin()