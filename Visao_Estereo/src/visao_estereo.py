# Autores: Matheus Teixeira de Sousa (teixeira.sousa@aluno.unb.br)
#          João Luiz Machado Júnior (180137158@aluno.unb.br)
# Disciplina: Princípios de Visão Computacional - turma A

import cv2 as cv
import numpy as np
import os.path
import functions as f

def first_requirement():
	# Organize paths to find necessary images and calibration data
	base = os.path.abspath(os.path.dirname(__file__))
	if os.name == 'nt':
		base_new = base.replace('\\src', '')
	else:
		base_new = base.replace('/src', '')

	# Define vectors for images and their respective paths
	images = ['im0.png', 'im1.png']
	data = [os.path.join(base_new, 'data', 'Middlebury', 'Jadeplant-perfect'),
			os.path.join(base_new, 'data', 'Middlebury', 'Playtable-perfect')]
	name = ['Jadeplant', 'Playtable']

	for i in range (len(name)):
		print('\nLoading images from ' + name[i] + ' data base...', flush=True)
		imgL = cv.imread(os.path.join(data[i], images[0]), cv.IMREAD_GRAYSCALE)
		imgR = cv.imread(os.path.join(data[i], images[1]), cv.IMREAD_GRAYSCALE)

		calib_data = f.data_reader(os.path.join(data[i], 'calib.txt'))
		req = 1

		min_disp = int(calib_data[24])
		max_disp = int(calib_data[25])
		block = 3

		imgL = cv.copyMakeBorder(imgL, None, None, max_disp, None, cv.BORDER_CONSTANT)
		imgR = cv.copyMakeBorder(imgR, None, None, max_disp, None, cv.BORDER_CONSTANT)

		print('Calculating disparity map...', flush=True)
		filteredImg = f.disparity_calculator(imgL, imgR, min_disp, max_disp, block, req, os.path.join(data[i],'disparidade.pgm'))

		focal_length = calib_data[0]
		base_line = calib_data[19]
		center_l = calib_data[2]
		center_r = calib_data[2]
		
		# Compute depth map and save it in a specific folder
		print('Calculating depth map...', flush=True)
		f.image_depth(filteredImg, focal_length, base_line, center_l, center_r, req, os.path.join(data[i],'profundidade.png'))

def second_requirement():
	# Organize paths to find necessary images and calibration data
	base = os.path.abspath(os.path.dirname(__file__))
	base_new = base.replace('\\src', '')

	# Define vectors for images and their respective paths
	images = ['MorpheusL.jpg', 'MorpheusR.jpg']
	data = os.path.join(base_new, 'data', 'FurukawaPonce')

	print('\nLoading Morpheus images from FurukawaPonce data base...', flush=True)
	imgL = cv.imread(os.path.join(data, images[0]))
	imgR = cv.imread(os.path.join(data, images[1]))

	calib_dataL = f.data_reader(os.path.join(data, 'MorpheusL.txt'))
	calib_dataR = f.data_reader(os.path.join(data, 'MorpheusR.txt'))
	
	req = 2
	imgL = imgL[0:1200, 0:1200]
	block = 1
	max_disp = 16*18
	
	new_imgL, new_imgR, base_line = f.warp_images(imgL, imgR, calib_dataL, calib_dataR, req)

	new_imgL = cv.copyMakeBorder(new_imgL, None, None, max_disp, None, cv.BORDER_CONSTANT)
	new_imgR = cv.copyMakeBorder(new_imgR, None, None, max_disp, None, cv.BORDER_CONSTANT)

	# Compute disparity map
	print('Calculating disparity map...', flush=True)
	filteredImg = f.disparity_calculator(new_imgL, new_imgR, 0, max_disp, block, req, os.path.join(data,'disparidade.pgm'))

	focal_length = (calib_dataL[0] + calib_dataL[1])/2
	center_l = calib_dataL[2]
	center_r = calib_dataR[2]

	# Compute depth map and save it in a specific folder
	print('Calculating depth map...', flush=True)
	f.image_depth(filteredImg, focal_length, base_line, center_l, center_r, req, os.path.join(data,'profundidade.png'))

def third_requirement():
	# Organize paths to find necessary images and calibration data
	base = os.path.abspath(os.path.dirname(__file__))
	base_new = base.replace('\\src', '')

	# Define vectors for images and their respective paths
	images = ['MorpheusL.jpg', 'MorpheusR.jpg']
	data = os.path.join(base_new, 'data', 'FurukawaPonce')

	# Verify if the disparity map and accompanying rectified images exist;
	# if not, go back to requirement 2 and create them before continuing
	if os.path.isfile(os.path.join(data,'disparidade.pgm')) == False:
		print('Disparity map not found!', flush=True)
		second_requirement()

	imgL = cv.imread(os.path.join(data, images[0]))
	imgR = cv.imread(os.path.join(data, images[1]))

	calib_dataL = f.data_reader(os.path.join(data, 'MorpheusL.txt'))
	calib_dataR = f.data_reader(os.path.join(data, 'MorpheusR.txt'))
	
	req = 3
	imgL = imgL[0:1200, 0:1200]

	left_img, right_img, base_line, matrixP_L, matrixP_R = f.warp_images(imgL, imgR, calib_dataL, calib_dataR, req)

	# Show user images to collect depth, width and height measurements
	depth_and_height = f.lateral_measurements(right_img, base_line)
	width = f.frontal_measurement(left_img)

	P = np.array([[depth_and_height[0][0], depth_and_height[0][1], depth_and_height[1][0], depth_and_height[1][1]],
						[depth_and_height[2][0], depth_and_height[2][1], depth_and_height[3][0], depth_and_height[3][1]],
						[width[0][0], width[0][1], width[1][0], width[1][1]]])
	
	# Create matrix with 3D world coordinates to measure IRL distances
	gray_left_img = cv.cvtColor(left_img, cv.COLOR_BGR2GRAY)
	real_world_coordinates = 0.0239 * (f.world_coordinates(gray_left_img, base_line, matrixP_L, matrixP_R))

	f.box_size(P, real_world_coordinates)
	f.show_clicks(left_img, right_img, P)

if __name__ == "__main__":
	# Main menu for 
	data = input('Define the number of requirement (1, 2, 3): ')
	requirement = ['1', '2', '3']

	while data not in requirement: 
		print('\n\nError! There is no requirement number ' + data)
		data = input('Define the number of requirement (1, 2, 3): ')
	if data == '1':
		first_requirement()
	elif data == '2':
		second_requirement()
	elif data == '3':
		third_requirement()
