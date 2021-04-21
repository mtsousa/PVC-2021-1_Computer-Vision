# Autor: Matheus Teixeira de Sousa
# E-mail: teixeira.sousa@aluno.unb.br
# Disciplina: Princípios de Visão Computacional
#
# Captura o vídeo da webcam e substitui, em tempo real, o fundo predefinido
# por um background virtual também predefinido.

import cv2 as cv
import numpy as np

# Define o valor de algumas variáveis de interesse
PRETO = np.array([0],np.uint8)
BRANCO = np.array([255],np.uint8)
MINIMO = np.array([25],np.uint8)

# Captura o video da câmera e define a imagem para o background virtual
camera = cv.VideoCapture(0)
virtual_background = cv.imread('marvel_avengers_08.jpg')

# Captura o fundo para a imagem de referência
while(True): 
	resultado, imagem_referencia = camera.read()  
	cv.imshow('Imagem de fundo', imagem_referencia)

    # Define a saída do laço de repetição caso a tecla 'c' seja pressionada
	key = cv.waitKey(1) & 0xFF
	if ord('c') == key:
		break

cv.destroyAllWindows()

while(True):
    # Lê o vídeo capturado pela câmera e redimensiona o virtual background
    resultado, video = camera.read()
    virtual_background = cv.resize(virtual_background, (imagem_referencia.shape[1], imagem_referencia.shape[0]), interpolation = cv.INTER_AREA)

    # Realiza a diferença absoluta entre o video e a imagem de referência
    # Converte o resultado para uma escala em cinza
    imagem_dif = cv.absdiff(video, imagem_referencia)
    imagem_dif = cv.cvtColor(imagem_dif.astype(np.uint8), cv.COLOR_BGR2GRAY)

    # Verfica os valores dos pixels do vídeo e subtitui os valores maiores
    # por branco (255) e os menores ou iguais por preto (0)
    mascara = np.where(imagem_dif > MINIMO, BRANCO, PRETO)

    # Inverte o resultado anterior
    mascara_invert = cv.bitwise_not(mascara)

    # Soma a mascara com o video e a mascara_invert com o virtual_background
    foreground = cv.bitwise_and(video, video, mask = mascara)
    background = cv.bitwise_and(virtual_background, virtual_background, mask = mascara_invert)

    # Combina foregroung com background e apresenta o resultado
    imagem_final = cv.add(foreground, background)
    cv.imshow('Background Virtual', imagem_final)
    cv.imshow('Normal', video)

    # Define a saída do laço de repetição caso a tecla 'q' seja pressionada
    key = cv.waitKey(1) & 0xFF
    if ord('q') == key:
	    break

camera.release()
cv.destroyAllWindows()