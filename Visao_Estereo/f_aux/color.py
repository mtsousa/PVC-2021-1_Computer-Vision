# Autores: Matheus Teixeira de Sousa (teixeira.sousa@aluno.unb.br)
#          João Luiz Machado Júnior (180137158@aluno.unb.br)
# Disciplina: Princípios de Visão Computacional - turma A

import cv2 as cv
import matplotlib.pyplot as plt
import os.path

base = os.path.abspath(os.path.dirname(__file__))
if os.name == 'nt':
    base_new = base.replace('\\f_aux', '')
else:
    base_new = base.replace('/f_aux', '')

# Define o vetor da imagem e dos caminhos para a imagem
image = 'disparidade.pgm'
data = [os.path.join(base_new, 'data', 'Middlebury', 'Jadeplant-perfect'),
        os.path.join(base_new, 'data', 'Middlebury', 'Playtable-perfect')]

for i in range(len(data)):
    img = cv.imread(os.path.join(data[i], image), cv.IMREAD_UNCHANGED)

    # Mostra imagem de disparidades com mapa de cores, padrão "jet"
    plt.imshow(img, cmap='jet')
    plt.colorbar()
    plt.show()