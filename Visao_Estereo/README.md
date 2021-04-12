# Stereo Vision

### *Requirements:*
      1 - Depth map estimate from rectified stereo images: Calculation of disparity and depth map for two groups of images from Middlebury image base, PlayTable and JadePlant;

      2 - Stereo cameras with convergence: Calculation of disparity and depth map for another group of image, Morpheus action figure, from Furukawa and Ponce 3D Photography Dataset;

      3 - Minimum box: Determing the minimum box dimensions in which the object of requeriment 2 can fit.

### OpenCV version: 4.5.1
### Python version: 3.8.1

### Python modules used:
     - cv2 (contrib version) 
     - numpy  
     - os

### To install opencv contrib version with pip installer run the command:
>pip install opencv-contrib-python

### To evaluate the requeriments run the command:
- For Windows users:
>python src/visao_stereo.py
- For Linux users:
>python3 src/visao_stereo.py

### Sample of image in README
![alt text](https://www.serpro.gov.br/menu/noticias/noticias-2020/o-que-eh-visao-computacional/imagem1artigo.png/@@images/eeb269b6-1014-4539-a03d-946d931d535f.png)

### Other sample of image
![alt text](data/FurukawaPonce/profundidade.png)