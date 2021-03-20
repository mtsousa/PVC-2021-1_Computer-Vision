# Autores: Matheus Teixeira de Sousa (teixeira.sousa@aluno.unb.br)
#          João Luiz Machado Júnior (180137158@aluno.unb.br)
# Disciplina: Princípios de Visão Computacional - turma A

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import re

def data_reader(file_name):
# Função para leitura de dados de calibração em arquivo .txt

    lines = []
    data = []

    with open(file_name) as f:
        lines = f.readlines() # Armazenar linhas do arquivo em lista

    for i in range(len(lines)): data.append(re.findall(r"[-+]?\d*\.\d+|\d+", lines[i]))
    # Coleta linha-a-linha de números relevantes na lista

    del data[0][0]; # Remoção dos números irrelevantes que identificam o nome de cada câmera
    del data[1][0];

    return data

calib_jade_data = []
calib_table_data = []
imgL = cv.imread('im0.png', cv.COLOR_BGR2GRAY)
imgR = cv.imread('im1.png', cv.COLOR_BGR2GRAY)

window_size = 3

left_matcher = cv.StereoSGBM_create(
    minDisparity=-1,
    numDisparities=40*16,  # Numero maximo de disparidades (640 para essa imagem)
    blockSize=window_size,
    P1=8 * 3 * window_size,
    P2=32 * 3 * window_size,
    disp12MaxDiff=12,
    uniquenessRatio=10,
    speckleWindowSize=50,
    speckleRange=32,
    preFilterCap=63,
    mode=cv.STEREO_SGBM_MODE_SGBM_3WAY
)
right_matcher = cv.ximgproc.createRightMatcher(left_matcher)

# parâmetros do filtro
lmbda = 80000
sigma = 1.3
visual_multiplier = 6

wls_filter = cv.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
wls_filter.setLambda(lmbda)

wls_filter.setSigmaColor(sigma)
displ = left_matcher.compute(imgL, imgR) 
dispr = right_matcher.compute(imgR, imgL)
displ = np.int16(displ)
dispr = np.int16(dispr)
filteredImg = wls_filter.filter(displ, imgL, None, dispr)

# Normaliza o filtro
filteredImg = cv.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv.NORM_MINMAX)
filteredImg = np.uint8(filteredImg)

# cv.imshow('filtered', filteredImg) Corrigir o tamanho da janela de exibição
cv.imwrite('filtered.jpg', filteredImg)

# Utiliza matplotlib para criar perfil em pseudocor com barra de cores para a imagem
plt.imshow(filteredImg, cmap='jet')
plt.colorbar()
plt.savefig("color_filtered.jpg")
cv.waitKey(0)

calib_jade_data = data_reader('calib_jade.txt')
calib_table_data = data_reader('calib_table.txt')
