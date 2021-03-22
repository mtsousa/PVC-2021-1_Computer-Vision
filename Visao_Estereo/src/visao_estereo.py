# Autores: Matheus Teixeira de Sousa (teixeira.sousa@aluno.unb.br)
#          João Luiz Machado Júnior (180137158@aluno.unb.br)
# Disciplina: Princípios de Visão Computacional - turma A

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os.path

def data_reader(file_name):
# Função para leitura de dados de calibração em arquivo .txt

	lines = []
	data = []

	with open(file_name) as f:
		lines = f.readlines()  # Armazenar linhas do arquivo em lista

	# Percorrer linhas separando termos relevantes para inserir na lista
	for i in range(len(lines)):
		clean_line = lines[i].replace('=', ' ').replace('[', '').replace(';', '').replace(']', '')
		results = clean_line.split()
		data.append(results)

	# Remover strings que identificam o nome de cada variavel
	for i in range(len(lines)):
		del data[i][0]

	return data

def world_coordinates (img, calib_data):
# Função que utiliza dados de calibração das câmeras utilizadas para obter coordenadas 
# 3D dos pontos no espaço

	cx = float(calib_data[0][2])
	cy = float(calib_data[0][5])
	f = float(calib_data[0][0])
	bline = float(calib_data[3][0])
	doff = float(calib_data[2][0])

	Q = np.float32([[1, 0, 0, -cx],
                    [0, 1, 0, -cy],
                    [0, 0, 0, f], 
                    [0, 0, -1/bline, doff/bline]])


	# Utilizamos a matriz Q para realizar uma reprojeção, convertendo pixels com
	# valor de disparidade na sua sequência correspondente de coordenadas [x, y, z]
	world_coordinates = cv.reprojectImageTo3D(img, Q)
	# print(world_coordinates)

	pass

def image_depth (img, calib_data):
# Produz um mapa de profundidade, originalmente em milímetros mas normalizado para a escala 0 - 254 em preto e branco
# para os objetos na imagem

	f = float(calib_data[0][0])
	bline = float(calib_data[3][0])
	doff = float(calib_data[2][0])
	
	Z = bline * f / (np.array(img) + doff)

	filtered_depth_image = cv.normalize(src=Z, dst=Z, beta=0, alpha=254, norm_type=cv.NORM_MINMAX)
	filtered_depth_image = np.uint8(filtered_depth_image)
	
	cv.imshow('DepthMap', filtered_depth_image)
	cv.imwrite(os.path.join(data[0],'DepthMap.jpg'), filtered_depth_image)

	# Podemos mapear os valores da imagem de profundidade de volta para unidades em milímetros
	# por um simples ajuste de escala:
	# original = np.array((filtered_depth_image - minimo) / float(maximo))

	pass

def disparity_calculator(left_image, right_image, disparities_num, min_num):
# Função que calcula mapa de disparidades dadas duas imagens estereo-retificadas

	window_size = 3

	left_matcher = cv.StereoSGBM_create(
	    minDisparity = min_num,
	    numDisparities = disparities_num,  # Numero maximo de disparidades (600 para essa imagem)
	    blockSize = window_size,
	    P1 = 8*3*window_size,
	    P2 = 32*3*window_size,
	    disp12MaxDiff = 12,
	    uniquenessRatio = 10,
	    speckleWindowSize = 50,
	    speckleRange = 32,
	    preFilterCap = 63,
	    mode = cv.STEREO_SGBM_MODE_SGBM_3WAY
	)
	right_matcher = cv.ximgproc.createRightMatcher(left_matcher)

	# parâmetros do filtro
	lmbda = 80000
	sigma = 1.3
	visual_multiplier = 6

	wls_filter = cv.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
	wls_filter.setLambda(lmbda)

	wls_filter.setSigmaColor(sigma)
	displ = left_matcher.compute(left_image, right_image) 
	dispr = right_matcher.compute(right_image, left_image)
	displ = np.int16(displ)
	dispr = np.int16(dispr)
	filteredImg = wls_filter.filter(displ, left_image, None, dispr)

	# Normaliza o filtro
	filteredImg = cv.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv.NORM_MINMAX)
	filteredImg = np.uint8(filteredImg)

	return filteredImg

# -------------------------------------------------------------------------------

# Define o diretório anterior ao diretório do programa
base = os.path.abspath(os.path.dirname(__file__))
base_new = base.replace('\\src', '')

# Define os vetores das imagens e dos caminhos para as imagens
images = ['im0.png', 'im1.png', 'MorpheusL.jpg', 'MorpheusR.jpg', 'warriorL.jpg', 'warriorR.jpg']
data = [os.path.join(base_new, 'data', 'Middlebury', 'Jadeplant-perfect'),
		os.path.join(base_new, 'data', 'Middlebury', 'Playtable-perfect'),
		os.path.join(base_new, 'data', 'FurukawaPonce')]

imgL = cv.imread(os.path.join(data[0], images[0]), cv.COLOR_BGR2GRAY)
imgR = cv.imread(os.path.join(data[0], images[1]), cv.COLOR_BGR2GRAY)

calib_jade_data = data_reader(os.path.join(data[0], 'calib.txt'))
calib_table_data = data_reader(os.path.join(data[1], 'calib.txt'))

disparities_num = int(calib_jade_data[9][0])
min_disp = int(calib_jade_data[8][0])

filteredImg = disparity_calculator(imgL, imgR, disparities_num, min_disp)

# Cria a imagem no diretório espeficidado pelo caminho, nesse
# caso, é o mesmo diretório da imagem que ele leu
#cv.imshow('filtered', filteredImg) Corrigir o tamanho da janela de exibição
cv.imwrite(os.path.join(data[0],'filtered.jpg'), filteredImg)

# Mostra imagem de disparidades com mapa de cores, padrão "jet"
plt.imshow(filteredImg, cmap='jet')
plt.colorbar()
plt.savefig(os.path.join(data[0], 'color_filtered.jpg'))
plt.show()

world_coordinates(filteredImg, calib_jade_data)
image_depth(filteredImg, calib_jade_data)

cv.waitKey(0)
