# Autores: Matheus Teixeira de Sousa (teixeira.sousa@aluno.unb.br)
#          João Luiz Machado Júnior (180137158@aluno.unb.br)
# Disciplina: Princípios de Visão Computacional - turma A

import cv2 as cv
import numpy as np
import os.path

# Class to contain information about user mouse clicks
class Capture_Click:
	clicks_number = int

	def __init__(self):
		self.initial = []
		self.final = []
		self.clicks_number = 0

	def click(self,event,x,y,flags,param):
		
		if event == cv.EVENT_LBUTTONDOWN:
			self.initial.clear()
			self.initial.append(x)
			self.initial.append(y)
			self.clicks_number = 1

			print('x,y for starting coordinates = ', self.initial, flush=True)

		if event == cv.EVENT_RBUTTONDOWN:
			self.final.clear()
			self.final.append(x)
			self.final.append(y)
			if self.clicks_number == 1:
				self.clicks_number += 1
			else:
				self.clicks_number = 1

			print('x,y for ending coordinates = ', self.final, flush=True)

		if self.clicks_number == 2:
			print('\nYou have collected your coordinates!', flush=True)
			print('Press ESC to finish, or keep clicking if you want to change them!\n', flush=True)
			self.clicks_number = 0

def show_image(img, w, h, win_name):
	cv.namedWindow(win_name, cv.WINDOW_NORMAL)
	cv.resizeWindow(win_name, (w, h))
	cv.imshow(win_name, img)
	cv.waitKey(0)
	cv.destroyAllWindows()

def lateral_measurements(image):
	# Shows the user a rectified view of toy Morpheus sitting in a chair
	# looking all gloomy and conspiratory, and asks for two pairs of clicks to measure
	# the object's depth and height.

	coordinates = []

	depth_measurement = Capture_Click()
	cv.namedWindow('Morpheus (right view)', cv.WINDOW_NORMAL)
	cv.resizeWindow('Morpheus (right view)', (539, 431))

	cv.imshow('Morpheus (right view)', image)
	print('\n\n\t\tMeasuring your image\'s depth!', flush=True)
	print('Left click to set your starting coordinates and right click to set the end coordinates for your image depth!', flush=True)
	cv.setMouseCallback('Morpheus (right view)', depth_measurement.click)

	while(1):
	    cv.imshow('Morpheus (right view)', image)
	    k = cv.waitKey(20) & 0xFF
	    if k == 27:
	        break
	cv.destroyAllWindows()

	coordinates.append(depth_measurement.initial)
	coordinates.append(depth_measurement.final)

	height_measurement = Capture_Click()
	cv.namedWindow('Morpheus (right view)', cv.WINDOW_NORMAL)
	cv.resizeWindow('Morpheus (right view)', (539, 431))

	cv.imshow('Morpheus (right view)', image)
	print('\n\n\t\tMeasuring your image\'s height!', flush=True)
	print('Left click to set your starting coordinates and right click to set the end coordinates for your image height!', flush=True)
	cv.setMouseCallback('Morpheus (right view)', height_measurement.click)

	while(1):
	    cv.imshow('Morpheus (right view)', image)
	    k = cv.waitKey(20) & 0xFF
	    if k == 27:
	        break
	cv.destroyAllWindows()

	coordinates.append(height_measurement.initial)
	coordinates.append(height_measurement.final)

	return coordinates

def frontal_measurement(image):
	# Shows the user a (different) rectified view of toy Morpheus sitting in a chair,
	# still looking scary and misterious, and asks for two clicks to measure
	# the object's width.

	coordinates = []

	width_measurement = Capture_Click()
	cv.namedWindow('Morpheus (left view)', cv.WINDOW_NORMAL)
	cv.resizeWindow('Morpheus (left view)', (539, 431))
	cv.imshow('Morpheus (left view)', image)

	print('\n\n\t\tMeasuring your image\'s width!', flush=True)
	print('Left click to set your starting coordinates and right click to set the end coordinates for your image depth!', flush=True)
	cv.setMouseCallback('Morpheus (left view)', width_measurement.click)

	while(1):
	    cv.imshow('Morpheus (left view)', image)
	    k = cv.waitKey(20) & 0xFF
	    if k == 27:
	        break
	cv.destroyAllWindows()

	coordinates.append(width_measurement.initial)
	coordinates.append(width_measurement.final)

	return coordinates

def common_origin(points):
# Function to re-organize object measurements so all three lines
# come from a single origin

	origin = (880, 800)
	height = abs(points[1][1] - points[1][3])
	width = abs(points[2][0] - points[2][2])

	x_depth = abs(points[0][0] - points[0][2])
	y_depth = abs(points[0][1] - points[0][3])

	reorganized_points = np.array([[origin[0], origin[1], (origin[0] - x_depth), (origin[1] + y_depth)],
									[origin[0], origin[1], origin[0], (origin[1] - height)],
									[origin[0], origin[1], (origin[0] - width), origin[1]]])

	print(reorganized_points)

	return reorganized_points

def show_clicks(imageL, imageR, points):
# Function to show in an image the user's clicks and subsequent measurements

	green = (0, 255, 0)
	i = 0
	j = 0

	while i < len(points):
		start = (points[i][0], points[i][1])
		end = (points[i][2], points[i][3])

		cv.circle(imageL, start, 3, green, -1)
		cv.circle(imageL, end, 3, green, -1)
		cv.line(imageL, start, end, green, 2)

		i += 1

	cv.namedWindow('User clicks', cv.WINDOW_NORMAL)
	cv.resizeWindow('User clicks', (539, 431))
	
	while(1):
	    cv.imshow('User clicks', imageL)
	    k = cv.waitKey(20) & 0xFF
	    if k == 27:
	        break
	cv.destroyAllWindows()

	reorganized_points = common_origin(points)

	while j < len(reorganized_points):
		start = (reorganized_points[j][0], reorganized_points[j][1])
		end = (reorganized_points[j][2], reorganized_points[j][3])

		cv.circle(imageR, start, 3, green, -1)
		cv.circle(imageR, end, 3, green, -1)
		cv.line(imageR, start, end, green, 2)

		j += 1

	cv.namedWindow('User clicks', cv.WINDOW_NORMAL)
	cv.resizeWindow('User clicks', (539, 431))
	
	while(1):
	    cv.imshow('User clicks', imageR)
	    k = cv.waitKey(20) & 0xFF
	    if k == 27:
	        break
	cv.destroyAllWindows()

def data_reader(file_name):
# Function to acquire information from .txt files about calibrated cameras

	lines = []
	data = []

	with open(file_name) as f:
		lines = f.readlines()  # Store all lines in a list

	# Search list line by line, separating relevant data from irrelevant characters
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

def factorize_projection(matrixP_L, matrixP_R):
# Function that factorize a projective matrix as P = K[R|t]
	Q_L = np.linalg.inv(matrixP_L[0:3, 0:3])
	Q_R = np.linalg.inv(matrixP_R[0:3, 0:3])

	U_L, B_L = np.linalg.qr(Q_L)
	U_R, B_R = np.linalg.qr(Q_R)

	matR_L = np.linalg.inv(U_L)
	vect_L = np.matmul(B_L, matrixP_L[0:3, 3])
	matK_L = np.linalg.inv(B_L)
	matK_L = matK_L/matK_L[2, 2]

	matR_R = np.linalg.inv(U_R)
	vect_R = np.matmul(B_R, matrixP_R[0:3, 3])
	matK_R = np.linalg.inv(B_R)
	matK_R = matK_R/matK_R[2, 2]

	return matK_L, matK_R

def world_coordinates (left_img, base_line, matrixP_L, matrixP_R):
# Function that applies calibration data to rectified images alongside 
# openCV's reprojectImageTo3D to find [x, y, z] world coordinates

	calib_dataL, calib_dataR = factorize_projection(matrixP_L, matrixP_R)

	print('\ncalib_dataL = ', calib_dataL)
	print('\ncalib_dataR = ', calib_dataR)
	
	f = (calib_dataL[0][0] + calib_dataL[1][1])/2
	cx0 = calib_dataL[0][2]
	cy0 = calib_dataL[1][2]
	cx1 = calib_dataR[0][2]
	doff = abs(cx1 - cx0)
	bline = float(base_line)

	Q = np.float32([[1, 0, 0, -cx0],
					[0, 1, 0, -cy0],
					[0, 0, 0, f], 
					[0, 0, -1/bline, doff/bline]])

	world_coordinate_map = cv.reprojectImageTo3D(left_img, Q)

	print ('world_coordinate_map has size', world_coordinate_map.size)

	return world_coordinate_map

def image_depth (img, focal, base_line, center_l, center_r, req, save_dir):
# Function to create a depth map, originally in milimeters but normalized to a
# 0 - 254 black and white scale

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

	# The depth image can be mapped back to its original values in milimeters
	# through a scale adjustment:
	# original = np.array((filtered_depth_image - lowest_value) / float(highest_value))
	#return filtered_depth_image

def disparity_calculator(left_image, right_image, min_num, max_num, block, req, save_dir):
# Function that takes a set of two stereo-rectified images to create a disparity map

	left_matcher = cv.StereoSGBM_create(
	    minDisparity = min_num,
	    numDisparities = 16*(max_num//16), # Maximum number of disparities 
	    blockSize = block,
	    P1 = 8*3*block,
	    P2 = 32*3*block,
	    disp12MaxDiff = -1, # Deactivated
	    uniquenessRatio = 10,
	    speckleWindowSize = 50,
	    speckleRange = 32,
	    preFilterCap = 63,
	    mode = cv.STEREO_SGBM_MODE_SGBM_3WAY
	)
	right_matcher = cv.ximgproc.createRightMatcher(left_matcher)

	# filter parameters
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

	# Calculate the rotation and translation vectors
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

# Autores: Matheus Teixeira de Sousa (teixeira.sousa@aluno.unb.br)
#          João Luiz Machado Júnior (180137158@aluno.unb.br)
# Disciplina: Princípios de Visão Computacional - turma A

import cv2 as cv
import numpy as np
import os.path

# Class to contain information about user mouse clicks
class Capture_Click:
	clicks_number = int

	def __init__(self):
		self.initial = []
		self.final = []
		self.clicks_number = 0

	def click(self,event,x,y,flags,param):
		
		if event == cv.EVENT_LBUTTONDOWN:
			self.initial.clear()
			self.initial.append(x)
			self.initial.append(y)
			self.clicks_number = 1

			print('x,y for starting coordinates = ', self.initial, flush=True)

		if event == cv.EVENT_RBUTTONDOWN:
			self.final.clear()
			self.final.append(x)
			self.final.append(y)
			if self.clicks_number == 1:
				self.clicks_number += 1
			else:
				self.clicks_number = 1

			print('x,y for ending coordinates = ', self.final, flush=True)

		if self.clicks_number == 2:
			print('\nYou have collected your coordinates!', flush=True)
			print('Press ESC to finish, or keep clicking if you want to change them!\n', flush=True)
			self.clicks_number = 0

def show_image(img, w, h, win_name):
	cv.namedWindow(win_name, cv.WINDOW_NORMAL)
	cv.resizeWindow(win_name, (w, h))
	cv.imshow(win_name, img)
	cv.waitKey(0)
	cv.destroyAllWindows()

def lateral_measurements(image):
	# Shows the user a rectified view of toy Morpheus sitting in a chair
	# looking all gloomy and conspiratory, and asks for two pairs of clicks to measure
	# the object's depth and height.

	coordinates = []

	depth_measurement = Capture_Click()
	cv.namedWindow('Morpheus (right view)', cv.WINDOW_NORMAL)
	cv.resizeWindow('Morpheus (right view)', (539, 431))

	cv.imshow('Morpheus (right view)', image)
	print('\n\n\t\tMeasuring your image\'s depth!', flush=True)
	print('Left click to set your starting coordinates and right click to set the end coordinates for your image depth!', flush=True)
	cv.setMouseCallback('Morpheus (right view)', depth_measurement.click)

	while(1):
	    cv.imshow('Morpheus (right view)', image)
	    k = cv.waitKey(20) & 0xFF
	    if k == 27:
	        break
	cv.destroyAllWindows()

	coordinates.append(depth_measurement.initial)
	coordinates.append(depth_measurement.final)

	height_measurement = Capture_Click()
	cv.namedWindow('Morpheus (right view)', cv.WINDOW_NORMAL)
	cv.resizeWindow('Morpheus (right view)', (539, 431))

	cv.imshow('Morpheus (right view)', image)
	print('\n\n\t\tMeasuring your image\'s height!', flush=True)
	print('Left click to set your starting coordinates and right click to set the end coordinates for your image height!', flush=True)
	cv.setMouseCallback('Morpheus (right view)', height_measurement.click)

	while(1):
	    cv.imshow('Morpheus (right view)', image)
	    k = cv.waitKey(20) & 0xFF
	    if k == 27:
	        break
	cv.destroyAllWindows()

	coordinates.append(height_measurement.initial)
	coordinates.append(height_measurement.final)

	return coordinates

def frontal_measurement(image):
	# Shows the user a (different) rectified view of toy Morpheus sitting in a chair,
	# still looking scary and misterious, and asks for two clicks to measure
	# the object's width.

	coordinates = []

	width_measurement = Capture_Click()
	cv.namedWindow('Morpheus (left view)', cv.WINDOW_NORMAL)
	cv.resizeWindow('Morpheus (left view)', (539, 431))
	cv.imshow('Morpheus (left view)', image)

	print('\n\n\t\tMeasuring your image\'s width!', flush=True)
	print('Left click to set your starting coordinates and right click to set the end coordinates for your image depth!', flush=True)
	cv.setMouseCallback('Morpheus (left view)', width_measurement.click)

	while(1):
	    cv.imshow('Morpheus (left view)', image)
	    k = cv.waitKey(20) & 0xFF
	    if k == 27:
	        break
	cv.destroyAllWindows()

	coordinates.append(width_measurement.initial)
	coordinates.append(width_measurement.final)

	return coordinates

def common_origin(points):
# Function to re-organize object measurements so all three lines
# come from a single origin

	origin = (880, 800)
	height = abs(points[1][1] - points[1][3])
	width = abs(points[2][0] - points[2][2])

	x_depth = abs(points[0][0] - points[0][2])
	y_depth = abs(points[0][1] - points[0][3])

	reorganized_points = np.array([[origin[0], origin[1], (origin[0] - x_depth), (origin[1] + y_depth)],
									[origin[0], origin[1], origin[0], (origin[1] - height)],
									[origin[0], origin[1], (origin[0] - width), origin[1]]])

	return reorganized_points

def show_clicks(imageL, imageR, points):
# Function to show in an image the user's clicks and subsequent measurements

	green = (0, 255, 0)
	i = 0
	j = 0

	while i < len(points):
		start = (points[i][0], points[i][1])
		end = (points[i][2], points[i][3])

		cv.circle(imageL, start, 3, green, -1)
		cv.circle(imageL, end, 3, green, -1)
		cv.line(imageL, start, end, green, 2)

		i += 1

	cv.namedWindow('User clicks', cv.WINDOW_NORMAL)
	cv.resizeWindow('User clicks', (539, 431))
	
	while(1):
	    cv.imshow('User clicks', imageL)
	    k = cv.waitKey(20) & 0xFF
	    if k == 27:
	        break
	cv.destroyAllWindows()

	reorganized_points = common_origin(points)

	while j < len(reorganized_points):
		start = (reorganized_points[j][0], reorganized_points[j][1])
		end = (reorganized_points[j][2], reorganized_points[j][3])

		cv.circle(imageR, start, 3, green, -1)
		cv.circle(imageR, end, 3, green, -1)
		cv.line(imageR, start, end, green, 2)

		j += 1

	cv.namedWindow('User clicks', cv.WINDOW_NORMAL)
	cv.resizeWindow('User clicks', (539, 431))
	
	while(1):
	    cv.imshow('User clicks', imageR)
	    k = cv.waitKey(20) & 0xFF
	    if k == 27:
	        break
	cv.destroyAllWindows()

def data_reader(file_name):
# Function to acquire information from .txt files about calibrated cameras

	lines = []
	data = []

	with open(file_name) as f:
		lines = f.readlines()  # Store all lines in a list

	# Search list line by line, separating relevant data from irrelevant characters
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

def factorize_projection(matrixP_L, matrixP_R):
# Function that factorize a projective matrix as P = K[R|t]
	Q_L = np.linalg.inv(matrixP_L[0:3, 0:3])
	Q_R = np.linalg.inv(matrixP_R[0:3, 0:3])

	U_L, B_L = np.linalg.qr(Q_L)
	U_R, B_R = np.linalg.qr(Q_R)

	matR_L = np.linalg.inv(U_L)
	vect_L = np.matmul(B_L, matrixP_L[0:3, 3])
	matK_L = np.linalg.inv(B_L)
	matK_L = matK_L/matK_L[2, 2]

	matR_R = np.linalg.inv(U_R)
	vect_R = np.matmul(B_R, matrixP_R[0:3, 3])
	matK_R = np.linalg.inv(B_R)
	matK_R = matK_R/matK_R[2, 2]

	return matK_L, matK_R

def world_coordinates (left_img, base_line, matrixP_L, matrixP_R):
# Function that applies calibration data to rectified images alongside 
# openCV's reprojectImageTo3D to find [x, y, z] world coordinates

	calib_dataL, calib_dataR = factorize_projection(matrixP_L, matrixP_R)
	
	f = (calib_dataL[0][0] + calib_dataL[1][1])/2
	cx0 = calib_dataL[0][2]
	cy0 = calib_dataL[1][2]
	cx1 = calib_dataR[0][2]
	doff = abs(cx1 - cx0)
	bline = float(base_line)

	Q = np.float32([[1, 0, 0, -cx0],
					[0, 1, 0, -cy0],
					[0, 0, 0, f], 
					[0, 0, -1/bline, doff/bline]])

	world_coordinate_map = cv.reprojectImageTo3D(left_img, Q)

	return world_coordinate_map

def image_depth (img, focal, base_line, center_l, center_r, req, save_dir):
# Function to create a depth map, originally in milimeters but normalized to a
# 0 - 254 black and white scale

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

	# The depth image can be mapped back to its original values in milimeters
	# through a scale adjustment:
	# original = np.array((filtered_depth_image - lowest_value) / float(highest_value))
	#return filtered_depth_image

def disparity_calculator(left_image, right_image, min_num, max_num, block, req, save_dir):
# Function that takes a set of two stereo-rectified images to create a disparity map

	left_matcher = cv.StereoSGBM_create(
	    minDisparity = min_num,
	    numDisparities = 16*(max_num//16), # Maximum number of disparities 
	    blockSize = block,
	    P1 = 8*3*block,
	    P2 = 32*3*block,
	    disp12MaxDiff = -1, # Deactivated
	    uniquenessRatio = 10,
	    speckleWindowSize = 50,
	    speckleRange = 32,
	    preFilterCap = 63,
	    mode = cv.STEREO_SGBM_MODE_SGBM_3WAY
	)
	right_matcher = cv.ximgproc.createRightMatcher(left_matcher)

	# filter parameters
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

	# Calculate the rotation and translation vectors
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

	aux = np.matmul(-R, c1).reshape(3,1)
	new_extL = np.hstack((R, aux))
	new_extR = new_extL
	Pn1 = np.matmul(A_1, new_extL)
	Pn2 = np.matmul(A_2, new_extR)

	H1 = np.matmul(Pn1[:,0:3], np.linalg.inv(matrixP_L[:, 0:3]))
	H2 = np.matmul(Pn2[:,0:3], np.linalg.inv(matrixP_R[:, 0:3]))

	return H1, H2, Pn1, Pn2, np.linalg.norm(v1)

def warp_images(imgL, imgR, calibL, calibR, req):
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

	if req != 3:
		show_image(concat, 1000, 400, 'rectified')

	gray_ret1 = cv.cvtColor(ret1, cv.COLOR_BGR2GRAY)
	gray_ret2 = cv.cvtColor(ret2, cv.COLOR_BGR2GRAY)

	if req == 3:
		return ret1, ret2, baseline, matrixP_L, matrixP_R
	else:
		return gray_ret1, gray_ret2, baseline
