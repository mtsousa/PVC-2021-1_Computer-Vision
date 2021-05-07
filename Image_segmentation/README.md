# Image Segmentation

### *Requirements:*
    1 - 

### Tensorflow version: 1.15.0
### Python 3

### Python modules used:
    - cv2 
    - tensorflow 
    - numpy

### To install tensorflow version with pip installer run the command:
>pip install tensorflow==1.15.0 tensorflow-gpu==1.15.0

### Configurations of training set:
    - GPU:
    - NVIDIA:
    - Python:

### To evaluate the code run the command:
- For Windows users:
>python src/main.py
- For Linux users:
>python3 src/main.py

### Results of image segmentation:
![alt text]()
![alt text]()

### Fine-tuning configuration:

|     Parameter     |               Configurations              |
|:-----------------:|:-----------------------------------------:|
|   Training mode   | Stochastic Gradient Descent with Momentum |
|     Batch size    |                     2                     |
|   Learning rate   |           [0.01, 0.001, 0.0001]*          |
| Momentum learning |                    0.9                    |
|       Epoch       |                [5, 10, 15]*               |
|    Dataset size   |            [7000, 10500, 14000]*          |