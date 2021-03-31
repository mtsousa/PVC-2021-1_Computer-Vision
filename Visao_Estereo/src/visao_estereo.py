# Autores: Matheus Teixeira de Sousa (teixeira.sousa@aluno.unb.br)
#          João Luiz Machado Júnior (180137158@aluno.unb.br)
# Disciplina: Princípios de Visão Computacional - turma A

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os.path
import functions as f


def first_requirement():
	# Define o diretório anterior ao diretório do programa
	base = os.path.abspath(os.path.dirname(__file__))
	if os.name == 'nt':
		base_new = base.replace('\\src', '')
	else:
		base_new = base.replace('/src', '')

	# Define os vetores das imagens e dos caminhos para as imagens
	images = ['im0.png', 'im1.png']
	data = [os.path.join(base_new, 'data', 'Middlebury', 'Jadeplant-perfect'),
			os.path.join(base_new, 'data', 'Middlebury', 'Playtable-perfect')]
	name = ['Jadeplant', 'Playtable']

	for i in [0,1]:
		print('\nLoading images from ' + name[i] + ' data base...', flush=True)
		imgL = cv.imread(os.path.join(data[i], images[0]), cv.IMREAD_GRAYSCALE)
		imgR = cv.imread(os.path.join(data[i], images[1]), cv.IMREAD_GRAYSCALE)

		calib_data = f.data_reader(os.path.join(data[i], 'calib.txt'))

		min_disp = int(calib_data[8][0])
		max_disp = int(calib_data[9][0])

		print('Calculating disparity map...', flush=True)
		filteredImg = f.disparity_calculator(imgL, imgR, min_disp, max_disp)
		
		# Redimensiona a imagem para uma melhor visualização
		cv.namedWindow('filtered', cv.WINDOW_NORMAL)
		cv.resizeWindow('filtered', (439, 331))

		# Mostra o resultado do mapa de disparidade e o salva no diretório especificado
		cv.imshow('filtered', filteredImg)
		cv.waitKey(0)
		cv.destroyAllWindows()
		cv.imwrite(os.path.join(data[i],'disparidade.pgm'), filteredImg)

		# Calcula o mapa de profundidade e o salva no diretório especificado
		print('Calculating depth map...', flush=True)
		f.image_depth(filteredImg, calib_data, os.path.join(data[i],'profundidade.png'))

def second_requirement():
	# Define o diretório anterior ao diretório do programa
	base = os.path.abspath(os.path.dirname(__file__))
	base_new = base.replace('\\src', '')

	# Define os vetores das imagens e dos caminhos para as imagens
	images = ['MorpheusL.jpg', 'MorpheusR.jpg']
	data = os.path.join(base_new, 'data', 'FurukawaPonce')

	print('\nLoading Morpheus images from FurukawaPonce data base...', flush=True)
	imgL = cv.imread(os.path.join(data, images[0]), cv.IMREAD_GRAYSCALE)
	imgR = cv.imread(os.path.join(data, images[1]), cv.IMREAD_GRAYSCALE)

	imgL = f.resize_image(imgL, imgR)
	new_imgL, new_imgR = f.image_rectify(imgL, imgR)

	# Calcula o mapa de disparidade e de profundidade #

	cv.namedWindow('imgL', cv.WINDOW_NORMAL)
	cv.resizeWindow('imgL', (439, 331))
	cv.namedWindow('imgR', cv.WINDOW_NORMAL)
	cv.resizeWindow('imgR', (439, 331))

	cv.imshow('imgL', new_imgL)
	cv.imshow('imgR', new_imgR)
	cv.waitKey(0)
	cv.destroyAllWindows()

	#f.world_coordinates(filteredImg, calib_data)

def third_requirement():
	pass

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