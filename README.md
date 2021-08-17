# Autoencoder
```python
#import
from matplotlib.pyplot import imshow
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Sequential

#visualize the for loop
from tqdm import tqdm

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')

#take image path
img_path =  "flower_images/"

#Append to list all image name 
import os
img_path_list = os.listdir(img_path) 

#Preprocessing 

img_data = []

for i in tqdm(img_path_list):
    path = os.path.join(img_path ,i)
    img = cv2.imread(path,1)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #to see in original view
    img = img.astype('float32') / 255.     # all images are needed to convert between 0 and 1
    img=cv2.resize(img,(128, 128))
    img_data.append(img)
    #output
    

#Transform to the gray format , 
gray_image = []

for i in tqdm(img_data):
    img = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) 
    gray_image.append(img)

#show input and output 
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 4))
for i in range(10):
    # display original
    ax = plt.subplot(3, 20, i + 1)
    plt.imshow(img_data[i], cmap="binary")
    
    # display gray
    ax = plt.subplot(3, 20, 20 +i+ 1)
    plt.imshow(gray_image[i], cmap="binary")
    

#Reshape the trainable format
output_data = np.reshape(img_data, (len(img_data), 128, 128, 3))
input_data = np.reshape(gray_image, (len(gray_image), 128, 128, 1))


#Build Model
model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(128, 128, 1)))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
 

model.add(MaxPooling2D((2, 2), padding='same'))
 
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(3, (3, 3), activation='relu', padding='same'))

model.compile(optimizer='adam', loss='mse',metrics=['accuracy'])

model.summary()

#Fit model 


model.fit(input_data, output_data,
        validation_split=0.1,
        epochs=10000,
        batch_size=16,
        shuffle=True)

#Test Image
test_img = cv2.imread("monalisa.jpg",0)
test_img = test_img.astype('float32') / 255.     # all images are needed to convert between 0 and 1
test_img=cv2.resize(test_img,(128, 128))
imshow(test_img, cmap="gray")
#model input
test_img = test_img.reshape(1,128,128,1)
pred = model.predict(test_img)
#show predicted image
imshow(pred[0].reshape(128,128,3), cmap="binary")

#Save Model
model.save('imagecolorization_autoencoder.model')
```

**for any problem , don't hesitate to contact me from** [Linkedin](https://www.linkedin.com/in/mntalhakilic/) :+1: :+1:
