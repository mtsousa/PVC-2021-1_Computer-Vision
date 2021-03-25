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
	base_new = base.replace('\\src', '')

	# Define os vetores das imagens e dos caminhos para as imagens
	images = ['im0.png', 'im1.png']
	data = [os.path.join(base_new, 'data', 'Middlebury', 'Jadeplant-perfect'),
			os.path.join(base_new, 'data', 'Middlebury', 'Playtable-perfect')]
	name = ['Jadeplant', 'Playtable']

	for i in [0,1]:
		print('\nLoading images from ' + name[i] + ' data base...', flush=True)
		imgL = cv.imread(os.path.join(data[i], images[0]), cv.COLOR_BGR2GRAY)
		imgR = cv.imread(os.path.join(data[i], images[1]), cv.COLOR_BGR2GRAY)

		calib_data = f.data_reader(os.path.join(data[i], 'calib.txt'))
		#calib_table_data = f.data_reader(os.path.join(data[1], 'calib.txt'))

		disparities_num = int(calib_data[9][0])
		min_disp = int(calib_data[8][0])
		
		print('Calculating disparity map...', flush=True)
		filteredImg = f.disparity_calculator(imgL, imgR, disparities_num, min_disp)

		# Redimensiona a imagem para uma melhor visualização
		cv.namedWindow('filtered', cv.WINDOW_NORMAL)
		cv.resizeWindow('filtered', (439, 331))

		# Cria a imagem no diretório espeficidado pelo caminho, nesse
		# caso, é o mesmo diretório da imagem que ele leu
		cv.imshow('filtered', filteredImg)
		cv.waitKey(0)
		cv.destroyAllWindows()
		cv.imwrite(os.path.join(data[i],'filtered.jpg'), filteredImg)

		# Mostra imagem de disparidades com mapa de cores, padrão "jet"
		# plt.imshow(filteredImg, cmap='jet')
		# plt.colorbar()
		# plt.savefig(os.path.join(data[i], 'color_filtered.jpg'))
		# plt.show()

		print('Calculating depth map...', flush=True)
		f.image_depth(filteredImg, calib_data, os.path.join(data[i],'DepthMap.jpg'))

def second_requirement():
	# Define o diretório anterior ao diretório do programa
	base = os.path.abspath(os.path.dirname(__file__))
	base_new = base.replace('\\src', '')

	# Define os vetores das imagens e dos caminhos para as imagens
	images = ['MorpheusL.jpg', 'MorpheusR.jpg', 'warriorL.jpg', 'warriorR.jpg']
	data = os.path.join(base_new, 'data', 'FurukawaPonce')

	# f.world_coordinates(filteredImg, calib_data)

def third_requirement():
	pass

if __name__ == "__main__":
	# Apenas uma ideia de interação com o usuário para definição do dado do projeto
	data = input('Define the number of requirement (1, 2, 3): ')
	requirement = ['1', '2', '3']

	while data not in requirement: 
		print('\n\nThere is not requiremente number ' + data)
		data = input('Define the number of requirement (1, 2, 3): ')
	if data == '1':
		first_requirement()
	elif data == '2':
		second_requirement()
	elif data == '3':
		third_requirement()