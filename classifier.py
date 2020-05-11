import tensorflow as tf
from tensorflow.keras import models
import os
from PIL import Image
import numpy as np
import pandas as pd
keras = tf.keras

IMG_SIZE = 160
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
IMG_PATH = 'images'

def format_example(image):
    image = tf.cast(image, tf.float32)
    image = (image/127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image

model = tf.keras.models.load_model('dogs_vs_cats.h5')

def prep_data():
    n = len(os.listdir(IMG_PATH))
    train_data = np.empty((n, 160, 160, 3))
    index = []
    i=0
    for img in os.listdir(IMG_PATH):
        image = Image.open(os.path.join(IMG_PATH, img))
        image = np.array(image)
        image = format_example(image)
        train_data[i] = image
        index.append(img.split('.')[0])
        i+=1
    return train_data, index

data, index = prep_data()   
a = model.predict(data)

n = len(os.listdir(IMG_PATH))
result = ['Cat'] * n
for i in range(n):
    j = int(index[i])-1
    if a[i]>=0:
        result[j] = 'Dog'


final = pd.DataFrame({'ImageId':[(i+1) for i in range(n)], 'Prediction':result})
final.to_csv('Output.csv', index=False)