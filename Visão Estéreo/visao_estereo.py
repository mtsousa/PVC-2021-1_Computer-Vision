# Autores: Matheus Teixeira de Sousa (teixeira.sousa@aluno.unb.br)
#          João Luiz Machado Júnior (180137158@aluno.unb.br)
# Disciplina: Princípios de Visão Computacional - turma A

import cv2 as cv
import numpy as np

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
cv.waitKey(0)