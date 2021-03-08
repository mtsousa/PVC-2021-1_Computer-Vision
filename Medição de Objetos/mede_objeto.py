# Autor: Matheus Teixeira de Sousa
# E-mail: teixeira.sousa@aluno.unb.br
# Disciplina: Princípios de Visão Computacional
#
# Mensura a altura ou a distância de um objeto considerando, respectivamente,
# uma distância ou uma altura conhecida e utilizando a distância focal da câmera
# obtida por meio da calibração por padrão de tabuleiro de xadrez.

import cv2 as cv
import glob

# calcula a distância do objeto até a câmera
def measure_distance(focalLength, height_know, height_px):
    return (height_know * focalLength) / height_px

# calcula a altura do objeto
def measure_height(focalLength, distance_know, height_px):
    return (distance_know * height_px) / focalLength

# Lê o arquivo de calibração gerado pelo programa calibra.py
# e retorna a média entre as distâncias focais f_x e f_y
def focal_length():
    ref_arquivo = open('dados_câmera.txt', 'r')
    linha = ref_arquivo.readline()
    for i in range(2):
        linha = ref_arquivo.readline()
        valores = linha.split()
        if i == 0:
            aux1 = valores[0]
            aux2 = aux1.split('[[')
            f_x = aux2[1] 
        else:
            f_y = valores[2]
    focalLength = (float(f_x) + float(f_y))/2       
    ref_arquivo.close()
    return focalLength

# Escreve o resultado da medida na imagem e apresenta
# o resultado juntamente com o valor esperado
def result(img, points, result, data):
    image = img.copy()
    # Escreve o valor obtido
    cv.line(image, points[0], points[1], (255,255,0), 3)
    cv.putText(image, "%.2fcm" % result,
		(image.shape[1] - 220, image.shape[0] - 20), cv.FONT_HERSHEY_SIMPLEX,
		1.5, (0, 255, 0), 3)
    # Escreve o valor esperado
    cv.putText(image, "%.2fcm" % data,
		(image.shape[1] - 610, image.shape[0]-20), cv.FONT_HERSHEY_SIMPLEX,
		1.5, (0, 255, 255), 3)
    cv.imshow("image", image)
    cv.waitKey(0)

# Captura dos eventos do botão esquerdo do mouse e escreve 
# as coordenadas do clique na imagem e no vetor 'points'
points = []
def click_event(event, x, y, flags, params): 
	global points
	# Procura por cliques no lado esquerdo do mouse 
	if event == cv.EVENT_LBUTTONDOWN: 
		points.append((x,y))
		# Mostra a coordenada do clique na imagem
		font = cv.FONT_HERSHEY_SIMPLEX 
		cv.putText(img_copy, str(x) + ',' +
					str(y), (x,y), font, 
					0.5, (0, 255, 255), 2) 
		cv.imshow('image', img_copy)

# Define a altura e as distâncias conhecidas em cm
height_know = 9.5
distance_know = [20, 30, 40, 60]

flag = 0
key = '0'
focalLength = focal_length() # Define a distância focal 

while(key != 'h' and key != 'd' and key != 'q'):
    key = input('Define the measure (\'h\' to HEIGHT or \'d\' to DISTANCE): ')

if key != 'q':
    path = glob.glob('.\images\*.jpg')
    for img in path:
        image = cv.imread(img)
        img_copy = image.copy()

        # Inicia a captura dos eventos do mouse 
        cv.imshow('image', img_copy) 
        cv.setMouseCallback('image', click_event)  
        cv.waitKey(0)
        
        # Calcula a altura do objeto
        height_px = abs(points[0][1] - points[1][1])

        # Calcula a altura ou a distância do objeto e
        # apresenta o resultado na imagem
        if key == 'h':
            height = measure_height(focalLength, distance_know[flag], height_px)
            result(image, points, height, height_know)
            flag += 1
        elif key == 'd':
            distance = measure_distance(focalLength, height_know, height_px)
            result(image, points, distance, distance_know[flag])
            flag += 1
        points.clear()