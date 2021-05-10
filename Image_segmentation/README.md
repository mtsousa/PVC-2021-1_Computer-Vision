# Image Segmentation

One of the most significant applications of computer vision is the parsing of objects with correct contextual identification. In this project deep learning strategies and tools are applied in conjuction with Microsoft's Common Objects in Context ([MS COCO](https://cocodataset.org/#home)) image dataset to explore the identification and segmentation of objects and beings in images. A convolutional neural network is fine-tuned and applied to test images with the ultimate objective of demonstrating on a practical level object detection for tasks such as automated navigation.

## Model

This is an implementation of [Mask R-CNN](https://arxiv.org/abs/1703.06870) on Python 3, Keras, and TensorFlow developed by Waleed Abdulla in [Matterport](https://github.com/matterport/Mask_RCNN) GitHub repository. The model generates bounding boxes and segmentation masks for each instance of an object in the image. It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone

## Setup the Environment

    - Python 3
    - Tensorflow version: 1.15.0
    - Keras version: 2.1.0

### Python modules used:
    - opencv-python 
    - tensorflow
    - keras 
    - numpy
    - argparse
    - scipy
    - Pillow
    - cython
    - matplotlib
    - scikit-image
    - tensorflow
    - keras
    - h5py
    - imgaug
    - IPython
    - pycocotools

### To install all modules run the command:
>pip install -r requirements.txt

## Usage 

### Training detail:
* This model was trained on Google Colab, which has 12GB of RAM memory available, NVIDIA Tesla T4 GPU, and 30GB of disk memory;
* The [pre-trained weights](https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5) used were trained on MS COCO dataset;
* This model detect and segment **7 categories** of the original MS COCO dataset: person, dog, car, motorcycle, bicycle, bus, and truck;
* The dataset used had 7000 images for training, **1000 images for each category**, and 500 images for validation from 1 to 40 epochs and 10500 images for training, **1500 images for each category**, and 3056 images for validation for more 20 epochs;
* Each epoch has 1000 iterations and the learning rate was 0.001 for all epochs;
* Were applied data augmentation via horizontal flip, vertical flip, and rotation for 50% of images.

### To download the images to training run the command [1]:
- For Windows users:
    >python f_aux/save_images.py train

    >python f_aux/save_images.py val
- For Linux users:
    >python3 f_aux/save_images.py train

    >python3 f_aux/save_images.py val

[1]: It will download 10500 images for training and 3056 images for validation. The annotations file to train and to evaluate will be download automatically after run one of those commands. 

### To train the model run the command:
- For Windows users:
    >python src/train_model.py train --dataset=path/to/images --annotations=path/to/annotations --classes=7 --model=path/to/model
- For Linux users:
    >python3 src/train_model.py train --dataset=path/to/images --annotations=path/to/annotations --classes=7 --model=path/to/model

### To evalute the model on MS COCO metric run the command:
- For Windows users:
    >python src/train_model.py evaluate --dataset=path/to/images --annotations=path/to/annotations --classes=7 --model=path/to/model
- For Linux users:
    >python3 src/train_model.py evaluate --dataset=path/to/images --annotations=path/to/annotations --classes=7 --model=path/to/model

### To apply inference mode run the command [2]:
- For Windows users:
>python src/main.py --image=path/to/image.jpg
- For Linux users:
>python3 src/main.py --image=path/to/image.jpg

[2]: The model weights will be download automatically after run this command.

## Segmentation results

### Visual results
![alt text](images/colagem_readme.PNG)

### Average precision

| 5 epochs | 10 epochs | 15 epochs | 20 epochs | 25 epochs | 30 epochs | 35 epochs | 40 epochs | 45 epochs | 50 epochs | 55 epochs[3] | 60 epochs |
|:--------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
|   0.320  |   0.341   |   0.342   |   0.343   |   0.343   |   0.354   |   0.342   |   0.351   |   0.347   |   0.363   |   0.367   |   0.359   |

[3]: Our best result was **0.368** on 54th epoch.
