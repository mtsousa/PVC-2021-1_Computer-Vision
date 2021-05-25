# -*- coding: utf-8 -*-
# Autores: Matheus Teixeira de Sousa (teixeira.sousa@aluno.unb.br)
#          João Luiz Machado Júnior (180137158@aluno.unb.br)
# Disciplina: Princípios de Visão Computacional - turma A

import cv2 as cv
import argparse
import matplotlib
import urllib.request
import shutil
import os
import tensorflow as tf

# Import Mask RCNN
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
import matplotlib.pyplot as plt

class InferenceConfig(Config):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    NAME = 'pvc_net'
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Our classes + background
    NUM_CLASSES = 1 + 7

# Directory to save logs and trained model
MODEL_DIR = 'src/logs'

# Parse command line arguments
parser = argparse.ArgumentParser(
	description='Made an inference on image.')
parser.add_argument('--image', required=True,
                        metavar="/path/to/image.jpg",
                        help='Directory to image to make inference')

# Directory to image
args = parser.parse_args() 
IMAGE_DIR = args.image

# Create model object in inference mode
config = InferenceConfig()
DEVICE = "/cpu:0"
with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

model_path = 'src/model/mask_rcnn_pvc_net_0050.h5'
MODEL_URL = 'https://github.com/mtsousa/Visao_Computacional/releases/download/v2_MyModel/mask_rcnn_pvc_net_0050.h5'

# Download weights if does not exists
if not os.path.exists(model_path):
    print('Donwloading model...')
    with urllib.request.urlopen(MODEL_URL) as resp, open(model_path, 'wb') as out:
        shutil.copyfileobj(resp, out)
# Load weights
model.load_weights(model_path, by_name=True)

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'dog']

# Read the image and convert from BGR to RGB
image = cv.imread(IMAGE_DIR)
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

# Run detection
results = model.detect([image], verbose=1)

# Visualize results
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            class_names, r['scores'])