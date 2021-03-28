# Autores: Matheus Teixeira de Sousa (teixeira.sousa@aluno.unb.br)
#          João Luiz Machado Júnior (180137158@aluno.unb.br)
# Disciplina: Princípios de Visão Computacional - turma A

import cv2 as cv
import numpy as np

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

def image_depth (img, calib_data, save_dir):
# Produz um mapa de profundidade, originalmente em milímetros mas normalizado para a escala 0 - 254 em preto e branco
# para os objetos na imagem

	f = float(calib_data[0][0])
	bline = float(calib_data[3][0])
	aux = np.zeros(img.shape)
	img_float = aux + img
	new_diff = img_float - aux
	new_diff[new_diff == 0.0] = np.inf
	
	Z = bline * f / new_diff
	filtered_depth_image = cv.normalize(src=Z, dst=Z, beta=0, alpha=254, norm_type=cv.NORM_MINMAX)
	filtered_depth_image = np.uint8(filtered_depth_image)
	filtered_depth_image[filtered_depth_image == 0] = 255

	# Redimensiona a imagem para uma melhor visualização
	cv.namedWindow('DepthMap', cv.WINDOW_NORMAL)
	cv.resizeWindow('DepthMap', (439, 331))

	cv.imshow('DepthMap', filtered_depth_image)
	cv.imwrite(save_dir, filtered_depth_image)
	cv.waitKey(0)
	cv.destroyAllWindows()
	# Podemos mapear os valores da imagem de profundidade de volta para unidades em milímetros
	# por um simples ajuste de escala:
	# original = np.array((filtered_depth_image - minimo) / float(maximo))

def disparity_calculator(left_image, right_image, min_num, max_num):
# Função que calcula mapa de disparidades dadas duas imagens estereo-retificadas

	window_size = 3

	left_matcher = cv.StereoSGBM_create(
	    minDisparity = min_num,
	    numDisparities = 16*(max_num//16), # Numero maximo de disparidades
	    blockSize = window_size,
	    P1 = 8*3*window_size**2,
	    P2 = 32*3*window_size**2,
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
