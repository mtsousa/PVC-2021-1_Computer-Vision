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
	data = os.path.join(base_new, 'data', 'FurukawaPonce')
	images = ['rectifiedL.jpg', 'rectifiedR.jpg']

	# Verify if the disparity map and accompanying rectified images exist;
	# if not, go back to requirement 2 and create them before continuing
	if os.path.isfile(os.path.join(data,'disparidade.pgm')) == False:
		print('Disparity map not found!')
		second_requirement()

	# Show user images to collect depth, width and height measurements
	left_img = cv.imread(os.path.join(data, images[0]))
	right_img = cv.imread(os.path.join(data, images[1]))

	depth_measurement = f.depth_by_clicks(right_img)
	height_and_width = f.cross_section(left_img)

	P = np.array([[depth_measurement[0][0], depth_measurement[0][1], depth_measurement[1][0], depth_measurement[1][1]],
						[height_and_width[0][0], height_and_width[0][1], height_and_width[1][0], height_and_width[1][1]],
						[height_and_width[2][0], height_and_width[2][1], height_and_width[3][0], height_and_width[3][1]]])

	print('\npoints =\n', P)
	
	# Create matrix with 3D world coordinates to measure IRL distances
	real_world_coordinates = 0.0239 * (f.world_coordinates(data, max_disp = 288))

	# Each element in the world coordinates matrix has values x, y and z.

	print('\n\nreal_world_coordinates at that spot =\n')
	print(real_world_coordinates[depth_measurement[0][0]][depth_measurement[0][1]], 'to', real_world_coordinates[depth_measurement[1][0]][depth_measurement[1][1]])

if __name__ == "__main__":
	# Apenas uma ideia de interação com o usuário para definição do dado do projeto
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
