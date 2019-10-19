#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import six.moves.urllib as urllib
import sys


# In[2]:



import tarfile
import tensorflow as tf


# In[3]:


import zipfile
import cv2


# In[4]:



from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image


# In[5]:


# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

from object_detection.utils import ops as utils_ops

from utils import label_map_util

from utils import visualization_utils as vis_util


# In[ ]:





# In[6]:


# Define the video stream
cap = cv2.VideoCapture(0)  # Change only if you have more than one webcams

# What model to download.
# Models can bee found here: 
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
MODEL_NAME = 'ssd_inception_v2_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')


# In[ ]:





# In[7]:



# Number of classes to detect
NUM_CLASSES = 300

# Download Model
opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
    file_name = os.path.basename(file.name)
    if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, os.getcwd())



# In[8]:


# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


# In[ ]:





# In[9]:


# Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`,
# we know that this corresponds to `airplane`.  Here we use internal utility functions, but
# anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# In[ ]:





# In[10]:




# Helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


# In[11]:


config = tf.ConfigProto()
config.gpu_options.allow_growth = True


# In[13]:



# Detection
with detection_graph.as_default():
    with tf.Session(graph=detection_graph, config=config) as sess:
        while True:

            # Read frame from camera
            ret, image_np = cap.read()
            
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            
            # Extract image tensor
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            
            # Extract detection boxes
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            
            # Extract detection scores
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            
            # Extract detection classes
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            
            # Extract number of detectionsd
            num_detections = detection_graph.get_tensor_by_name(
                'num_detections:0')
            
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)

            # Display output
            cv2.imshow('object detection', cv2.resize(image_np, (800, 600)))

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


import os


# In[ ]:




