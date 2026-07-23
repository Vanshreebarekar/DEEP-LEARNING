#!/usr/bin/env python
# coding: utf-8

# In[1]:


# only for cuda enabled laptop and desktop
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
if physical_devices:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)


# In[2]:


# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)
print(np.__version__)


# In[3]:


fashion_mnist=keras.datasets.fashion_mnist

(train_images,train_labels),(test_images,test_labels)=fashion_mnist.load_data()


# In[4]:


train_images.shape


# In[5]:


test_images.shape


# In[6]:


train_labels


# In[7]:


train_images[0]


# In[8]:


train_labels[0]


# In[9]:


train_images[0].shape


# In[10]:


plt.imshow(train_images[0],cmap='gray_r')
plt.xticks([])
plt.yticks([])
plt.show()


# In[11]:


plt.imshow(train_images[90],cmap='gray_r')
plt.xticks([])
plt.yticks([])
plt.show()


# In[12]:


plt.imshow(train_images[50],cmap='gray_r')
plt.title(train_labels[50])
plt.xticks([])
plt.yticks([])
plt.show()


# In[13]:


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'] # all 10 classes are include


# In[14]:


class_names


# In[15]:


train_labels[50]


# In[16]:


class_names[train_labels[50]]   # 1st train_labels=>3 then class_names[3]


# In[17]:


plt.imshow(train_images[50],cmap='gray_r')
plt.title(class_names[train_labels[50]])
plt.xticks([])
plt.yticks([])
plt.show()


# In[18]:


plt.imshow(train_images[150],cmap='gray_r')
plt.title(class_names[train_labels[150]])
plt.xticks([])
plt.yticks([])
plt.show()


# In[19]:


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)  # starting from 551 to 5525
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary) # color only white &black
    plt.xlabel(class_names[train_labels[i]])
plt.tight_layout()
plt.show()


# In[20]:


train_images = train_images / 255.0  # answer in 1 or 0

test_images = test_images / 255.0


# In[21]:


train_images[0]


# In[22]:


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(10,activation='softmax')
])


# In[23]:


model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


# In[24]:


history=model.fit(train_images,train_labels,validation_split=0.2,epochs=50)


# In[25]:


history.history


# In[26]:


history.history['loss']


# In[27]:


epochs = 50
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc=0)
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc=0)
plt.title('Training and Validation Loss')
plt.show()


# # early Stopping

# In[28]:


model_new = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(10,activation='softmax')
])


# In[29]:


model_new.compile(optimizer='adam',
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])


# In[30]:


callback = keras.callbacks.EarlyStopping(monitor='val_loss',patience=5)


# In[31]:


history = model_new.fit(train_images, train_labels, epochs=50, validation_split=0.2,callbacks=callback)


# In[33]:


epochs = 13
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc=0)
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc=0)
plt.title('Training and Validation Loss')
plt.show()


# In[34]:


plt.figure(figsize=(10,10))
plt.imshow(test_images[125],cmap='gray_r')
plt.title(class_names[test_labels[125]])
plt.xticks([])
plt.yticks([])
plt.show()


# In[35]:


model.predict(test_images[125].reshape(1,28,28))


# In[36]:


model.predict(test_images[125].reshape(1,28,28)).round()


# In[37]:


class_names


# In[38]:


np.argmax(model.predict(test_images[125].reshape(1,28,28)))


# In[39]:


class_names[np.argmax(model.predict(test_images[125].reshape(1,28,28)))]


# In[ ]:




