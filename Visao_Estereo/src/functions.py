# Autores: Matheus Teixeira de Sousa (teixeira.sousa@aluno.unb.br)
#          João Luiz Machado Júnior (180137158@aluno.unb.br)
# Disciplina: Princípios de Visão Computacional - turma A

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

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

def image_depth (img, focal, base_line, save_dir):
# Produz um mapa de profundidade, originalmente em milímetros mas normalizado para a escala 0 - 254 em preto e branco
# para os objetos na imagem

	f = focal
	bline = base_line
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

    height, width, _ = imgR.shape
    imgL = cv.resize(imgL, (width, height), interpolation = cv.INTER_LINEAR)
    return imgL

def intrinsic_matrix(calib):
    K = np.zeros((3,3))
    K[0,0] = calib[0]
    K[0,1] = calib[4]
    K[0,2] = calib[2]
    K[1,1] = calib[1]
    K[1,2] = calib[3]
    K[2,2] = 1
    return K

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

# Visualize epilines
# Adapted from: https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html
def drawlines(img1src, lines, pts1src):

	r, c, _ = img1src.shape
	img1color = img1src.copy()
	# Edit: use the same random seed so that two images are comparable!
	np.random.seed(0)
	i = 0
	for r, pt1 in zip(lines, pts1src):
		color = tuple(np.random.randint(0, 255, 3).tolist())
		x0, y0 = map(int, [0, -r[2]/r[1]])
		x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
		if i%6 == 0:
			img1color = cv.line(img1color, (x0, y0), (x1, y1), color, 3)
		img1color = cv.circle(img1color, tuple(pt1), 5, color, 5)
		i += 1
	return img1color

def image_rectify(imgL, imgR):
# Função que encontra os pontos de match e retifica as imagens

	imgL_name = 'imgL'
	imgR_name = 'imgR'

	# find the keypoints and descriptors with SIFT
	sift = cv.SIFT_create()
	kp1, des1 = sift.detectAndCompute(imgL, None)
	kp2, des2 = sift.detectAndCompute(imgR, None)

	# Visualize keypoints
	imgSift = cv.drawKeypoints(
		imgL, kp1, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	cv.namedWindow('SIFT Keypoints', cv.WINDOW_NORMAL)
	cv.resizeWindow('SIFT Keypoints', (700, 650))
	cv.imshow("SIFT Keypoints", imgSift)
	cv.waitKey(0)
	cv.destroyAllWindows()

	# Match keypoints in both images
	# Based on: https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
	FLANN_INDEX_KDTREE = 1
	index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
	search_params = dict(checks=20)   # or pass empty dictionary
	flann = cv.FlannBasedMatcher(index_params, search_params)
	matches = flann.knnMatch(des1, des2, k=2)

	# Keep good matches: calculate distinctive image features
	# Lowe, D.G. Distinctive Image Features from Scale-Invariant Keypoints. International Journal of Computer Vision 60, 91–110 (2004). https://doi.org/10.1023/B:VISI.0000029664.99615.94
	# https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
	matchesMask = [[0, 0] for i in range(len(matches))]
	good = []
	pts1 = []
	pts2 = []

	for i, (m, n) in enumerate(matches):
		if m.distance < 0.5*n.distance:
			# Keep this keypoint pair
			matchesMask[i] = [1, 0]
			good.append(m)
			pts2.append(kp2[m.trainIdx].pt)
			pts1.append(kp1[m.queryIdx].pt)

	# Calculate the fundamental matrix for the cameras
	# https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html
	pts1 = np.int32(pts1)
	pts2 = np.int32(pts2)
	fundamental_matrix, inliers = cv.findFundamentalMat(pts1, pts2, cv.FM_RANSAC)

	# We select only inlier points
	pts1 = pts1[inliers.ravel() == 1]
	pts2 = pts2[inliers.ravel() == 1]

	# Find epilines corresponding to points in right image (second image) and
	# drawing its lines on left image
	lines1 = cv.computeCorrespondEpilines(
		pts2.reshape(-1, 1, 2), 2, fundamental_matrix)
	lines1 = lines1.reshape(-1, 3)
	line_imgL = drawlines(imgL, lines1, pts1)

	# Find epilines corresponding to points in left image (first image) and
	# drawing its lines on right image
	lines2 = cv.computeCorrespondEpilines(
		pts1.reshape(-1, 1, 2), 1, fundamental_matrix)
	lines2 = lines2.reshape(-1, 3)
	line_imgR = drawlines(imgR, lines2, pts2)

	cv.namedWindow('left epipolar lines', cv.WINDOW_NORMAL)
	cv.resizeWindow('left epipolar lines', (439, 331))
	cv.namedWindow('right epipolar lines', cv.WINDOW_NORMAL)
	cv.resizeWindow('right epipolar lines', (439, 331))

	cv.imshow('left epipolar lines', line_imgL)
	cv.imshow('right epipolar lines', line_imgR)
	cv.waitKey(0)
	cv.destroyAllWindows()

	# Stereo rectification (uncalibrated variant)
	# Adapted from: https://stackoverflow.com/a/62607343
	h1, w1, _ = imgL.shape
	h2, w2, _ = imgR.shape
	_, H1, H2 = cv.stereoRectifyUncalibrated(
		np.float32(pts1), np.float32(pts2), fundamental_matrix, imgSize=(w1, h1)
	)

	# Undistort (rectify) the images and save them
	# Adapted from: https://stackoverflow.com/a/62607343
	imgL_rectified = cv.warpPerspective(imgL, H1, (w1, h1), cv.BORDER_ISOLATED)
	imgR_rectified = cv.warpPerspective(imgR, H2, (w2, h2), cv.BORDER_ISOLATED)

	y = 58
	x = 58
	h, w, _ = imgR_rectified.shape
	crop_imgR = imgR_rectified[y:h-y, x:w-x]
	crop_imgL = imgL_rectified[y:h-y, x:w-x]

	cv.namedWindow(imgL_name, cv.WINDOW_NORMAL)
	cv.resizeWindow(imgL_name, (439, 331))
	cv.namedWindow(imgR_name, cv.WINDOW_NORMAL)
	cv.resizeWindow(imgR_name, (439, 331))

	cv.imshow(imgL_name, imgL_rectified)
	cv.imshow(imgR_name, imgR_rectified)
	cv.waitKey(0)
	cv.destroyAllWindows()

	# cv.imwrite("rectified_1.jpg", imgL_rectified)
	# cv.imwrite("rectified_2.jpg", imgR_rectified)
	# cv.imwrite("cropped_imgL.jpg", crop_imgL)
	# cv.imwrite("cropped_imgR.jpg", crop_imgR)

	crop_imgR = cv.cvtColor(crop_imgR, cv.COLOR_BGR2GRAY)
	crop_imgL = cv.cvtColor(crop_imgL, cv.COLOR_BGR2GRAY)

	return crop_imgL, crop_imgR

def calculate_baseline(calibL, calibR):
	r_vecL, t_vecL = extrinsic_parameters(calibL)
	r_vecR, t_vecR = extrinsic_parameters(calibR)
	
	newR_vecL = np.zeros((1,3))
	newR_vecR = np.zeros((1,3))

	newt_vecL = newR_vecL
	newt_vecR = newR_vecR

	newt_vecL = np.array([t_vecL[0], t_vecL[1], t_vecL[2]])
	newt_vecR = np.array([t_vecR[0], t_vecR[1], t_vecR[2]])

	newR_vecL, _ = cv.Rodrigues(r_vecL)
	newR_vecR, _ = cv.Rodrigues(r_vecR)

	rvec, tvec, _, _, _, _, _, _, _, _, = cv.composeRT(newR_vecL.ravel(), newt_vecL.ravel(), newR_vecR.ravel(), newt_vecR.ravel())
	
	print('vet_rot: \n', rvec, flush=True)
	baseline = np.linalg.norm(tvec)

	return baseline