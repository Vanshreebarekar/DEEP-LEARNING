#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2


# In[2]:


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


# In[3]:


image = cv2.imread("C:\\Users\\NITRO\\project_work\\dog.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# In[4]:


plt.figure(figsize=(20,20))
plt.imshow(image)
plt.axis('off')
plt.show()


# In[5]:


pip install segment-anything


# In[6]:


from segment_anything import sam_model_registry, SamPredictor #specific "point-and-click" tasks 
model_type = "vit_h"  
device = "cpu" 
# Vision Transformer-huge
model_type = "vit_h" 

#This points to the 2.5GB weight file (.pth). Without this, the model has no "knowledge"
sam_checkpoint = "C:\\Users\\NITRO\\project_work\\sam_vit_h_4b8939 (1).pth"

# Then load the model as usual
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)


# In[11]:


import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

sam_checkpoint = "C:\\Users\\NITRO\\project_work\\sam_vit_h_4b8939 (1).pth"
model_type = "vit_h"


device = "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)


# In[12]:


masks = mask_generator.generate(image)


# In[13]:


print(len(masks))
print(masks[0].keys())


# In[14]:


plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show() 


# # Automatic mask generation options

# In[15]:


mask_generator_2 = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.86,#Intersection over Union.
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,  # Requires open-cv to run post-processing
)


# In[16]:


masks2 = mask_generator_2.generate(image)


# In[17]:


len(masks2)


# In[18]:


plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks2)
plt.axis('off')
plt.show() 


# In[ ]:




