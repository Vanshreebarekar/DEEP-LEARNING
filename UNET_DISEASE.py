#!/usr/bin/env python
# coding: utf-8

# In[1]:


#https://www.kaggle.com/datasets/hamzamohiuddin/isbi-2012-challenge


# In[2]:


import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers,Model


# In[3]:


get_ipython().system('unzip /content/archive.zip')


# In[4]:


image_dir = "C:\\Users\\NITRO\\NLP\\unmodified-data\\train\\imgs"
mask_dir = "C:\\Users\\NITRO\\NLP\\unmodified-data\\train\\labels"
img_size = (256, 256)


# In[5]:


images = []
masks = []

for filename in os.listdir(image_dir):
    img_path = os.path.join(image_dir, filename)
    mask_path = os.path.join(mask_dir, filename)

    if os.path.exists(mask_path):
        img = load_img(img_path, target_size=img_size, color_mode="grayscale")
        img = img_to_array(img) / 255.0

        mask = load_img(mask_path, target_size=img_size, color_mode="grayscale")
        mask = img_to_array(mask) / 255.0

        images.append(img)
        masks.append(mask)

X = np.array(images)
Y = np.array(masks)


# In[6]:


X.shape


# In[7]:


Y.shape


# In[8]:


plt.subplots(1, 2, figsize = (10, 6))
plt.subplot(1, 2, 1)
plt.imshow(X[0])
plt.title("Image")
plt.subplot(1, 2, 2)
plt.title("Mask")
plt.imshow(Y[0])
plt.show()


# In[9]:


X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[10]:


inputs = layers.Input(shape=(256, 256, 1))

# Encoder
c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c1)
p1 = layers.MaxPooling2D((2, 2))(c1)

c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c2)
p2 = layers.MaxPooling2D((2, 2))(c2)

# Bottleneck
c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c3)

# Decoder
u1 = layers.UpSampling2D((2, 2))(c3)
u1 = layers.Concatenate()([u1, c2])
c4 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u1)
c4 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c4)

u2 = layers.UpSampling2D((2, 2))(c4)
u2 = layers.Concatenate()([u2, c1])
c5 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(u2)
c5 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c5)

outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c5)

model = Model(inputs=[inputs], outputs=[outputs])


# In[11]:


model.summary()


# In[12]:


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[13]:


model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=200, batch_size=8)


# In[16]:


test_image_dir = "C:\\Users\\NITRO\\NLP\\unmodified-data\\test\\imgs"
test_mask_dir = "C:\\Users\\NITRO\\NLP\\unmodified-data\\test\\labels"
test_images = []
test_masks = []

for filename in os.listdir(test_image_dir):
    img_path = os.path.join(test_image_dir, filename)
    mask_path = os.path.join(test_mask_dir, filename)

    if os.path.exists(mask_path):
        img = load_img(img_path, target_size=img_size, color_mode="grayscale")
        img = img_to_array(img) / 255.0

        mask = load_img(mask_path, target_size=img_size, color_mode="grayscale")
        mask = img_to_array(mask) / 255.0

        test_images.append(img)
        test_masks.append(mask)

X_test = np.array(test_images)
y_test = np.array(test_masks)


# In[17]:


X_test.shape


# In[18]:


y_test.shape


# In[19]:


preds = model.predict(X_test[3].reshape(1, 256, 256, 1))


# In[20]:


preds.shape


# In[21]:


plt.subplots(1, 3, figsize = (10, 6))
plt.subplot(1, 3, 1)
plt.title("Input Image")
plt.imshow(X_test[3])
plt.subplot(1, 3, 2)
plt.imshow(preds.reshape(256, 256))
plt.title("Predicted Image (y_hat)")
plt.subplot(1, 3, 3)
plt.title("Original Image (y)")
plt.imshow(y_test[3])
plt.show()


# In[22]:


predictions = model.predict(X_test)
predictions = (predictions > 0.5).astype(np.uint8)  # Binarize predicted masks


# In[23]:


import matplotlib.pyplot as plt

n = 5  # Number of samples to show

for i in range(n):
    plt.figure(figsize=(12, 4))

    # Original Image
    plt.subplot(1, 3, 1)
    plt.imshow(X_test[i].squeeze(), cmap='gray')
    plt.title('Test Image')
    plt.axis('off')


    plt.subplot(1, 3, 2)
    plt.imshow(y_test[i].squeeze(), cmap='gray')
    plt.title('True Mask')
    plt.axis('off')

    # Predicted Mask
    plt.subplot(1, 3, 3)
    plt.imshow(predictions[i].squeeze(), cmap='gray')
    plt.title('Predicted Mask')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# In[ ]:




