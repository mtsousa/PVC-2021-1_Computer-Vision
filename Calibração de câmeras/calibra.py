# Autor: Matheus Teixeira de Sousa
# E-mail: teixeira.sousa@aluno.unb.br
# Disciplina: Princípios de Visão Computacional
#
# Programa baseado no artigo de Kaustubh Sadekar e Satya Mallick, disponível em 
# https://learnopencv.com/camera-calibration-using-opencv/
# 
# Calibra a webcam a partir de um conjunto de fotos de um tabuleiro de xadrez
# e fornece a matriz de projeção P, a matriz de parâmetros intrínsecos K, a matriz
# de rotação R, o vetor de translação t e os parâmetros de distorção radial.

#!/usr/bin/env python

import cv2 as cv
import numpy as np
import glob

# Função que escreve o arquivo com os parâmetros da câmera
def escreve_arquivo(arquivo, matrix_p, matrix_k, matrix_r, vect_t, distortion):
    arquivo.write('Matriz K: \n')
    arquivo.write(str(matrix_k))
    arquivo.write('\n\nMatriz R: \n')
    arquivo.write(str(matrix_r))
    arquivo.write('\n\nVetor t: \n')
    arquivo.write(str(vect_t))
    arquivo.write('\n\nMatriz P: \n')
    arquivo.write(str(matrix_p))
    arquivo.write('\n\nParâmetros de distorção: \n')
    arquivo.write(str(distortion))

# Define as dimensões do tabuleiro de xadrez
CHECKERBOARD = (7,9)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Cria os vetores para guardar vetores dos pontos 3D e 2D de cada imagem do tabuleiro
obj_points = []
img_points = [] 

# Define as coordenadas para os pontos 3D no mundo
world_obj_p = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
world_obj_p[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None

# Lê o pacote das imagens capturadas com final .jpg num diretório específico
images = glob.glob('*.jpg')
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    # Encontra os cantos do tabuleiro
    result, corners = cv.findChessboardCorners(gray, CHECKERBOARD, cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE)

    # Verifica se o número de cantos foi encontrado e, caso verdadeiro,
    # refina as coordenadas dos pixels e os exibe na imagem
    if result == True:
        obj_points.append(world_obj_p)
        # Refina a coordenada dos pixels para determinados pontos 2D
        corners_new = cv.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
        img_points.append(corners_new)

        # Exibe os cantos encontrados
        img = cv.drawChessboardCorners(img, CHECKERBOARD, corners_new, result)
    
        cv.imshow('imagem',img)
        cv.waitKey(500)

cv.destroyAllWindows()

# Calibra a câmera a partir dos pontos 3D conhecidos e das coordenadas correspondentes
# Retorna a matriz K, os parâmetros de distorção, o vetor t e o vetor de rotação R 
result, matrix_k, distortion, r_vect, t_vect = cv.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

# Converte o vetor de rotação R na matriz de rotação R
# Utiliza os dados da primeira imagem
rotation_mat = np.zeros(shape=(3, 3))
matrix_r = cv.Rodrigues(r_vect[0], rotation_mat)[0]

# Determina a matrix de projeção P a partir dos dados da primeira imagem
matrix_ext = np.column_stack((matrix_r, t_vect[0]))
matrix_p = np.matmul(matrix_k, matrix_ext)

# Cria e escreve o arquivo com os parâmetros obtidos
arquivo = open('dados_câmera.txt', 'w')
escreve_arquivo(arquivo, matrix_p, matrix_k, matrix_r, t_vect[0], distortion)
arquivo.close()
