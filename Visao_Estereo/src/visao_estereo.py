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

	for i in [0,1]:
		print('\nLoading images from ' + name[i] + ' data base...', flush=True)
		imgL = cv.imread(os.path.join(data[i], images[0]), cv.IMREAD_GRAYSCALE)
		imgR = cv.imread(os.path.join(data[i], images[1]), cv.IMREAD_GRAYSCALE)

		calib_data = f.data_reader(os.path.join(data[i], 'calib.txt'))

		min_disp = int(calib_data[24])
		max_disp = int(calib_data[25])

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

		focal_length = calib_data[0]
		base_line = calib_data[19]
		# Calcula o mapa de profundidade e o salva no diretório especificado
		print('Calculating depth map...', flush=True)
		f.image_depth(filteredImg, focal_length, base_line, os.path.join(data[i],'profundidade.png'))

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
	
	imgL = f.resize_image(imgL, imgR)
	new_imgL, new_imgR = f.image_rectify(imgL, imgR)

	# Calcula o mapa de disparidade e de profundidade #
	print('Calculating disparity map...', flush=True)
	filteredImg = f.disparity_calculator(new_imgL, new_imgR, 0, 16*3)

	# Redimensiona a imagem para uma melhor visualização
	cv.namedWindow('filtered', cv.WINDOW_NORMAL)
	cv.resizeWindow('filtered', (439, 331))

	# Mostra o resultado do mapa de disparidade e o salva no diretório especificado
	cv.imshow('filtered', filteredImg)
	cv.waitKey(0)
	cv.destroyAllWindows()
	#cv.imwrite(os.path.join(data,'disparidade.pgm'), filteredImg)

	# Calcula o mapa de profundidade e o salva no diretório especificado
	cam_translationL = [calib_dataL[14], calib_dataL[15], calib_dataL[16]]
	cam_translationR = [calib_dataR[14], calib_dataR[15], calib_dataR[16]]

	diff_vec = np.array(cam_translationL) - np.array(cam_translationR)
	base_line = np.linalg.norm(np.array(diff_vec))
	focal_length = (calib_dataL[0] + calib_dataL[1])/2

	print('Baseline for MorpheusL image can be estimated as: ', base_line, flush=True)
	
	baseline = f.calculate_baseline(calib_dataL, calib_dataR)
	print('Minha baseline: ', baseline, flush=True)
	
	print('Calculating depth map...', flush=True)
	f.image_depth(filteredImg, focal_length, base_line, os.path.join(data,'profundidade.png'))

	#f.world_coordinates(filteredImg, calib_data)

def third_requirement():
	# Define o diretório anterior ao diretório do programa
	base = os.path.abspath(os.path.dirname(__file__))
	base_new = base.replace('\\src', '')

	# Define os vetores da imagem e do caminho para a imagem
	image = 'MorpheusR.jpg'
	data = os.path.join(base_new, 'data', 'FurukawaPonce')

	img = cv.imread(os.path.join(data, image))

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