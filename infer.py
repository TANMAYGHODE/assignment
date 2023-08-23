import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import load_model
import os

model=load_model('Model.keras')
model.summary()

img_height = 224
img_width = 224
IMAGE_SIZE = [img_height, img_width]

class_names=[]

test_dir='/app/data/'
for i in sorted(os.listdir((test_dir))):
    class_names.append(i)

count=0
total_count=0

for i in sorted(os.listdir((test_dir))):
    for j in sorted(os.listdir(test_dir+i)):
      test_image_path=test_dir+i+'/'+j
      img = tf.keras.utils.load_img(test_image_path, target_size=(img_height, img_width))
      img_array = tf.keras.utils.img_to_array(img)
      img_array = tf.expand_dims(img_array, 0)
      predictions = model.predict(img_array)
      score = tf.nn.softmax(predictions[0])
      predicted_class=class_names[np.argmax(score)]
      total_count=total_count+1
      print(predicted_class,i)
      if predicted_class==i:
         count=count+1


print("Accuracy for the Model is: ",(count/total_count)*100)