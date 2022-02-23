# Imports
import glob
import json
import os
from os.path import splitext,basename
import uuid
import base64

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.get_logger().setLevel('ERROR')
from tensorflow import keras
from tensorflow.keras.models  import model_from_json
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import LabelEncoder

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec



# Processing functions
#######################################
# Loads a model given a specific path #
#######################################
def load_model(path = './models/SemImSeg_model_EfficientNetV2B0.h5' ):
    try:
        # path = splitext(path)[0]
        model = tf.keras.models.load_model(path)
        print("Model Loaded successfully...")
        return model
    except Exception as e:
        print(e)
            
# Load models
img_seg_net_path = "./models/SemImSeg_model_EfficientNetV2B0.h5"
model = load_model(img_seg_net_path)
print("[INFO] Model loaded successfully...")

#######
def load_image(datapoint):
  input_image = tf.image.resize(datapoint['image'], (128, 128))

  input_image = normalize(input_image)

  return input_image

def normalize(input_imagek):
  input_image = tf.cast(input_image, tf.float32) / 255.0

  return input_image

def load_image_org(datapoint):
  input_image = tf.image.resize(datapoint['image'], (128, 128))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

  input_image, input_mask = normalize(input_image, input_mask)

  return input_image, input_mask

def normalize_org(input_image, input_mask):
  # input_image = tf.cast(input_image, tf.float32) / 255.0
  input_mask -= 1
  return input_image, input_mask
######################################################################################
# Converts colors from BGR (as read by OpenCV) to RGB (so that we can display them), #
# also eventually resizes the image to fit the size the model has been trained on    #
######################################################################################
def preprocess_image(image_path,resize=True):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = img / 1.0
    # img = img / 255
    if resize:
        img = cv2.resize(img, (128,128))
    img = np.expand_dims(img, axis=0)    
    return img

#########################################################################
# Reconstructs the image from detected pattern into plate cropped image #
#########################################################################
def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    print (display_list[i].shape)
    try: 
        plt.imshow(tf.keras.utils.array_to_img(tf.squeeze(display_list[i],axis=0))) ## tensorflow 2.8
    except:
        plt.imshow(tf.keras.utils.array_to_img(display_list[i])) ## tensorflow 2.8
    #plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()


def display_org(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    ##plt.imshow(tf.keras.utils.array_to_img(display_list[i])) ## tensorflow 2.8
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()
def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]


def show_predictions(image,pred_mask):
  display([image, create_mask(pred_mask)])
    
    
def show_predictions_org(dataset=None, num=1):
  if dataset:
    for image, mask in dataset.take(num):
      pred_mask = model.predict(image)
      display([image[0], pred_mask[0], create_mask(pred_mask)])
  else:
    display([sample_image, sample_mask,
             create_mask(model.predict(sample_image[tf.newaxis, ...]))])


##############################################################################
# Returns the image of the car (vehicle) and the Licence plate image (LpImg) #
##############################################################################
def get_segmentation(image_path, Dmax=608, Dmin = 608):
    input_image = preprocess_image(image_path)
    pred_mask = model(input_image)
    
    show_predictions(input_image, pred_mask)
    return create_mask(pred_mask)



def process_file(filename):
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
        pred_mask = get_segmentation(filename)
    else:
        pred_mask = 'This is not a valid image file'
        input_image = 'none'
        print(pred_mask)
    return pred_mask


def process_base64_image(args_dict):
    img_type = args_dict.get('type') or 'jpg'
    base64img = args_dict.get('image')
    img_bytes = base64.decodebytes(base64img.encode())
    filename = os.path.join('/tmp', f'{uuid.uuid4()}.{img_type}')
    with open(filename, 'wb') as f:
        f.write(img_bytes)
    pred_mask = get_segmentation(filename)
    os.remove(filename)
    return pred_mask


def predict(args_dict):
    
    if args_dict.get('image') is not None:
        pred_mask = process_base64_image(args_dict)
    else:
        filename = os.path.join('dataset/images', args_dict.get('data'))
        print(filename)
        print("2")
        pred_mask = process_file(filename)
    return {'pred_mask': pred_mask.numpy().tolist()}