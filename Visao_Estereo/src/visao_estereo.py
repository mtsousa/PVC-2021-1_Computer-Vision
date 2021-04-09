# Autores: Matheus Teixeira de Sousa (teixeira.sousa@aluno.unb.br)
#          João Luiz Machado Júnior (180137158@aluno.unb.br)
# Disciplina: Princípios de Visão Computacional - turma A

import cv2 as cv
import numpy as np
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
		
		# Calcula o mapa de profundidade e o salva no diretório especificado
		print('Calculating depth map...', flush=True)
		f.image_depth(filteredImg, focal_length, base_line, center_l, center_r, req, os.path.join(data[i],'profundidade.png'))

def second_requirement():
	# Define o diretório anterior ao diretório do programa
	base = os.path.abspath(os.path.dirname(__file__))
	base_new = base.replace('\\src', '')

	# Define os vetores das imagens e dos caminhos para as imagens
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
	
	new_imgL, new_imgR, base_line = f.warp_images(imgL, imgR, calib_dataL, calib_dataR)

	new_imgL = cv.copyMakeBorder(new_imgL, None, None, max_disp, None, cv.BORDER_CONSTANT)
	new_imgR = cv.copyMakeBorder(new_imgR, None, None, max_disp, None, cv.BORDER_CONSTANT)

	# Calcula o mapa de disparidade e de profundidade
	print('Calculating disparity map...', flush=True)
	filteredImg = f.disparity_calculator(new_imgL, new_imgR, 0, max_disp, block, req, os.path.join(data,'disparidade.pgm'))

	focal_length = (calib_dataL[0] + calib_dataL[1])/2
	center_l = calib_dataL[2]
	center_r = calib_dataR[2]

	# Calcula o mapa de profundidade e o salva no diretório especificado
	print('Calculating depth map...', flush=True)
	f.image_depth(filteredImg, focal_length, base_line, center_l, center_r, req, os.path.join(data,'profundidade.png'))

def third_requirement():
	# Define o diretório anterior ao diretório do programa
	base = os.path.abspath(os.path.dirname(__file__))
	base_new = base.replace('\\src', '')

	# Define os vetores da imagem e do caminho para a imagem
	image = 'MorpheusR.jpg'
	data = os.path.join(base_new, 'data', 'FurukawaPonce')

	img = cv.imread(os.path.join(data, image))
	req = 3

	dimensions = 3 # Trocar o valor da variável para 0 após implementar o calculo dos pontos
	points = np.zeros((3,2))
	click = f.Capture_Click('image')
	result = []

	cv.imshow('image', img)
	while(dimensions < 3):
		if click.clicks_number > 0 :
			if click.clicks_number == 2:
				# Calcula a distância entre os dois pontos capturados
				# result = Recebe o valor calculado em mm
				# points[dimensions] = pontos ajustados da caixa
				click.clicks_number = 0
				dimensions += 1
				if dimensions == 3:
					print("Box size result in mm: ", result, flush=True)
					# Desenha a caixa na image do Morpheus	
					# Verifica se quer desenha nova caixa
					# Se sim, então dimensions = 0 e points.clear()
	cv.waitKey(0)

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