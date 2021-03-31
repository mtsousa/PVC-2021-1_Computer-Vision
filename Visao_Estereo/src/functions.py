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
	    P1 = 8*3*window_size,
	    P2 = 32*3*window_size,
	    disp12MaxDiff = -1, # Desabilitado 
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
	filteredImg = (wls_filter.filter(displ, left_image, None, dispr) / 16.0)

	filteredImg = cv.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv.NORM_MINMAX)
	filteredImg = np.uint8(filteredImg)
	
	return filteredImg

def resize_image(imgL, imgR):
# Função que redimensiona a imagem da esquerda

    height, width = imgR.shape
    imgL = cv.resize(imgL, (width, height), interpolation = cv.INTER_LINEAR)
    return imgL

def intrinsic_matrix(focal_length, princPoint, skew):
    K = np.zeros((3,3))
    K[0,0] = focal_length[0]
    K[0,1] = skew
    K[0,2] = princPoint[0]
    K[1,1] = focal_length[1]
    K[1,2] = princPoint[1]
    K[2,2] = 1
    return K

def extrinsic_matrix(R, Tc):
    Ext = np.zeros((3,4))
    Ext = np.array([[R[0,0], R[0,1], R[0,2], Tc[0]], 
				  [R[1,0], R[1,1], R[1,2], Tc[1]],
				  [R[2,0], R[2,1], R[2,2], Tc[2]]])
    return Ext

def image_rectify(img_left, img_right):
		
	fc = [[6704.926882, 6705.241311], [6682.125964, 6681.475962]] 
	cc = [[738.251932, 457.560286], [875.207200, 357.700292]]
	alpha = [0.000103, 0.000101]
	rotate_L = np.asarray([[0.70717199,  0.70613396, -0.03581348],
					[0.28815232, -0.33409066, -0.89741388], 
					[-0.64565936,  0.62430623, -0.43973369]])
	transl_L = np.array([-532.285900, 207.183600, 2977.408000])

	rotate_R = np.asarray([[0.48946344,  0.87099159, -0.04241701], 
					[0.33782142, -0.23423702, -0.91159734], 
					[-0.80392924,  0.43186419, -0.40889007]])
	transl_R = np.array([-614.549000, 193.240700, 3242.754000])

	matrixK_L = intrinsic_matrix(fc[0], cc[0], alpha[0])
	matrixK_R = intrinsic_matrix(fc[1], cc[1], alpha[1])

	matrix_ext_L = extrinsic_matrix(rotate_L, transl_L)
	matrix_ext_R = extrinsic_matrix(rotate_R, transl_R)

	matrixP_L = np.matmul(matrixK_L, matrix_ext_L)
	matrixP_R = np.matmul(matrixK_R, matrix_ext_R)
	
	h, w = img_left.shape

	mapL = cv.initUndistortRectifyMap(matrixK_L, (0,0,0,0,0), rotate_L, matrixP_L, (w,h), cv.CV_16SC2)
	mapR = cv.initUndistortRectifyMap(matrixK_R, (0,0,0,0,0), rotate_R, matrixP_R, (w,h), cv.CV_16SC2)

	inter = cv.INTER_LANCZOS4
	return cv.remap(img_left, mapL[0], mapL[1], inter), cv.remap(img_right, mapR[0], mapR[1], inter)