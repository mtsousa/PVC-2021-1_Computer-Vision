# Autores: Matheus Teixeira de Sousa (teixeira.sousa@aluno.unb.br)
#          João Luiz Machado Júnior (180137158@aluno.unb.br)
# Disciplina: Princípios de Visão Computacional - turma A

import cv2 as cv
import numpy as np

# Define a classe para a captura dos cliques
class Capture_Click:
    initial = ['', '']
    final = ['', '']
    clicks_number = int

    def __init__(self, name):
        cv.namedWindow(name, cv.WINDOW_GUI_NORMAL)
        cv.setMouseCallback(name, self.click)
        self.clicks_number = 0

    def click(self, event, x, y, flags, param):
		# Captura os cliques do lado esquerdo do mouse
        if event == cv.EVENT_LBUTTONDOWN:
            if self.clicks_number == 0:
                self.initial[0] = x
                self.initial[1] = y
                self.clicks_number += 1
                # Desenha o ponto capturado na imagem ou o escreve no terminal

            elif self.clicks_number == 1:
                self.final[0] = x
                self.final[1] = y
                self.clicks_number += 1
                # Desenha o ponto capturado na imagem ou o escreve no terminal

def show_image(img, w, h, win_name):
	cv.namedWindow(win_name, cv.WINDOW_NORMAL)
	cv.resizeWindow(win_name, (w, h))
	cv.imshow(win_name, img)
	cv.waitKey(0)
	cv.destroyAllWindows()

def data_reader(file_name):
# Função para leitura de dados de calibração em arquivo .txt

	lines = []
	data = []

	with open(file_name) as f:
		lines = f.readlines()  # Armazenar linhas do arquivo em lista

	# Percorrer linhas separando termos relevantes para inserir na lista
	for i in range(len(lines)):

		clean_line = lines[i].replace('=', ' ').replace('[', '').replace(';', '').replace(']', '').replace(',', '')
		str_results = clean_line.split()

		for j in range(len(str_results)):

			try : 
				float(str_results[j])
				data.append(float(str_results[j]))

			except :
				pass
	
	return data

def world_coordinates (img, calib_data):
# Função que utiliza dados de calibração das câmeras utilizadas para obter coordenadas 
# 3D dos pontos no espaço

	f = calib_data[0]
	cx = calib_data[2]
	cy = calib_data[5]
	bline =calib_data[19]
	doff = calib_data[18]

	Q = np.float32([[1, 0, 0, -cx],
                    [0, 1, 0, -cy],
                    [0, 0, 0, f], 
                    [0, 0, -1/bline, doff/bline]])

	# Utilizamos a matriz Q para realizar uma reprojeção, convertendo pixels com
	# valor de disparidade na sua sequência correspondente de coordenadas [x, y, z]
	world_coordinates = cv.reprojectImageTo3D(img, Q)

def image_depth (img, focal, base_line, center_l, center_r, req, save_dir):
# Produz um mapa de profundidade, originalmente em milímetros mas normalizado para a escala 0 - 254 em preto e branco
# para os objetos na imagem

	f = focal
	bline = base_line
	img[img == 0.0] = np.inf
	doff = abs(center_l - center_r)

	Z = bline * f / (img + doff)

	if req != 3:
		filtered_depth_image = cv.normalize(src=Z, dst=Z, beta=0, alpha=254, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
		filtered_depth_image[filtered_depth_image == 0] = 255
		show_image(filtered_depth_image, 439, 331, 'depth_map')
		cv.imwrite(save_dir, filtered_depth_image)
	else:
		return filtered_depth_image
	# Podemos mapear os valores da imagem de profundidade de volta para unidades em milímetros
	# por um simples ajuste de escala:
	# original = np.array((filtered_depth_image - minimo) / float(maximo))
	#return filtered_depth_image

def disparity_calculator(left_image, right_image, min_num, max_num, block, req, save_dir):
# Função que calcula mapa de disparidades dadas duas imagens estereo-retificadas

	left_matcher = cv.StereoSGBM_create(
	    minDisparity = min_num,
	    numDisparities = 16*(max_num//16), # Numero maximo de disparidades
	    blockSize = block,
	    P1 = 8*3*block,
	    P2 = 32*3*block,
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
	filteredImg[filteredImg == -1] = 0

	if req != 3:
		h, w = filteredImg.shape
		filteredImg = filteredImg[0:h, max_num:w]

		aux = cv.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
		show_image(aux, 439, 331, 'filtered')
		cv.imwrite(save_dir, aux)

		if req == 1:
			aux2 = cv.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
			new_dir = save_dir.replace('disparidade.pgm', '')
			new_dir = new_dir + 'disparidade.npy'
			np.save(new_dir, aux2)
	
	return filteredImg

def intrinsic_matrix(calib):
    K = np.zeros((3,3))
    dist = np.zeros((1,5))
    K = np.array([[calib[0], calib[4], calib[2]],
    			 [0., calib[1], calib[3]],
				 [0., 0., 1]])
    dist = np.array([calib[17], calib[18], calib[19], calib[20], calib[21]])
    return K, dist

def extrinsic_parameters(calib):
    r_vec = np.zeros((3,3))
    t_vec = np.zeros((3,1))
    r_vec = np.array([[calib[5], calib[6], calib[7]], 
				     [calib[8], calib[9], calib[10]],
		             [calib[11], calib[12], calib[13]]])
    t_vec = np.array([[calib[14]],
					 [calib[15]],	
					 [calib[16]]])
    
    return r_vec, t_vec

def stereo_rectify(calibL, calibR, d1, d2):

	# Calculate the instrinsic matrix
	matrixK_L, distL = intrinsic_matrix(calibL)
	matrixK_R, distR = intrinsic_matrix(calibR)

	# Calculate the rotation and translation vector relative
	r_vecL, t_vecL = extrinsic_parameters(calibL)
	r_vecR, t_vecR = extrinsic_parameters(calibR)
	
	extrinsic_L = np.column_stack((r_vecL, t_vecL))
	extrinsic_R = np.column_stack((r_vecR, t_vecR))

	matrixP_L = np.matmul(matrixK_L, extrinsic_L)
	matrixP_R = np.matmul(matrixK_R, extrinsic_R)

	c1 = np.matmul(np.transpose(r_vecL), np.matmul(np.linalg.inv(matrixK_L), matrixP_L[:,3]))
	c2 = np.matmul(np.transpose(r_vecR), np.matmul(np.linalg.inv(matrixK_R), matrixP_R[:,3]))

	v1 = (c1-c2)
	v2 = np.cross(np.transpose(r_vecL[2]),v1)
	v3 = np.cross(v1,v2)

	R = np.array([np.transpose(v1)/np.linalg.norm(v1), 
				  np.transpose(v2)/np.linalg.norm(v2), 
				  np.transpose(v3)/np.linalg.norm(v3)])

	A_1 = (matrixK_R + matrixK_L)/2
	A_1[0][1] = 0
	A_2 = A_1
	A_2[0][1] = 0   

	A_1[0][2] = A_1[0][2] + d1[0]
	A_1[1][2] = A_1[1][2] + d1[1]
	A_2[0][2] = A_2[0][2] + d2[0]
	A_2[1][2] = A_2[1][2] + d2[1]

	aux1 = np.matmul(-R, c1).reshape(3,1)
	aux2 = np.matmul(-R, c2).reshape(3,1)
	new_extL = np.hstack((R, aux1))
	new_extR = np.hstack((R, aux2))
	Pn1 = np.matmul(A_1, new_extL)
	Pn2 = np.matmul(A_2, new_extR)

	H1 = np.matmul(Pn1[: ,0: 3], np.linalg.inv(matrixP_L[:,0: 3]))
	H2 = np.matmul(Pn2[: ,0: 3], np.linalg.inv(matrixP_R[:,0: 3]))

	return H1, H2, Pn1, Pn2, np.linalg.norm(v1)

def warp_images(imgL, imgR, calibL, calibR):
	d1 = [0, 0]
	d2 = [0, 0]

	H1, H2, matrixP_L, matrixP_R, baseline = stereo_rectify(calibL, calibR, d1, d2)

	aux1 = [imgL.shape[0], imgL.shape[1], 1]
	aux2 = [imgR.shape[0], imgR.shape[1], 1]
	p_aux1 = np.matmul(H1, aux1)
	p_aux2 = np.matmul(H2, aux2)
	d1 = [aux1[0] - p_aux1[0]//aux1[2], aux1[1] - p_aux1[1]//aux1[2]]
	d2 = [aux2[0] - p_aux2[0]//aux2[2], aux2[1] - p_aux2[1]//aux2[2]]
	d1[1] = d2[1]

	H1, H2, matrixP_L, matrixP_R, baseline = stereo_rectify(calibL, calibR, d1, d2)

	ret1 = cv.warpPerspective(imgL, H1, (3000, 3000))
	ret2 = cv.warpPerspective(imgR, H2, (3000, 3000))

	ret1 = ret1[140:1340, 1800:3000]
	ret2 = ret2[134:1334, 0:1200]

	concat = cv.hconcat([ret1, ret2])
	show_image(concat, 1000, 400, 'rectified')

	ret1 = cv.cvtColor(ret1, cv.COLOR_BGR2GRAY)
	ret2 = cv.cvtColor(ret2, cv.COLOR_BGR2GRAY)

	return ret1, ret2, baseline