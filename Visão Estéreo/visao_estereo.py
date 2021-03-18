# Autores: Matheus Teixeira de Sousa (teixeira.sousa@aluno.unb.br)
#          João Luiz Machado Júnior (180137158@aluno.unb.br)
# Disciplina: Princípios de Visão Computacional - turma A

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

imgL = cv.imread('im0.png',0)
imgR = cv.imread('im1.png',0)
stereo = cv.StereoBM_create(numDisparities=640, blockSize=15)
#stereo = cv.StereoSGBM_create(numDisparities=640, blockSize=15)
disparity = stereo.compute(imgL,imgR)
plt.imshow(disparity,'gray')
plt.show()