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

		# Cria a imagem no diretório espeficidado pelo caminho, nesse
		# caso, é o mesmo diretório da imagem que ele leu
		cv.imshow('filtered', filteredImg)
		cv.waitKey(0)
		cv.destroyAllWindows()
		cv.imwrite(os.path.join(data[i],'disparidade.pgm'), filteredImg)
		
		# Mostra imagem de disparidades com mapa de cores, padrão "jet"
		plt.imshow(filteredImg, cmap='jet')
		plt.colorbar()
		# plt.savefig(os.path.join(data[i], 'color_filtered.jpg'))
		plt.show()

		# O valor máximo de disparidade pode ser retirado do arquivo de dados de calibração
		# ou aproximado como um terço da altura da imagem : (float(calib_data[5][0]) / 3)
		disp_max =  float(calib_data[9][0])
		gt = cv.imread(os.path.join(data[i], 'disp0-n.pgm'), cv.IMREAD_GRAYSCALE)
		deu_certo = f.taxa_erro(filteredImg, gt, disp_max)
		print('Taxa de pixels ruins: {:.2f}%'.format(deu_certo*100.0))

		print('Calculating depth map...', flush=True)
		f.image_depth(filteredImg, calib_data, os.path.join(data[i],'profundidade.png'))

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
		print('\n\nError! There is no requirement number ' + data)
		data = input('Define the number of requirement (1, 2, 3): ')
	if data == '1':
		first_requirement()
	elif data == '2':
		second_requirement()
	elif data == '3':
		third_requirement()
		
