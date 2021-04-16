# Autores: Matheus Teixeira de Sousa (teixeira.sousa@aluno.unb.br)
#          João Luiz Machado Júnior (180137158@aluno.unb.br)
# Disciplina: Princípios de Visão Computacional - turma A

import cv2 as cv
import numpy as np
import os.path

def evaldisp(disparity, gtdisp, badthresh):

	gt_shape = gtdisp.shape
	disp_shape = disparity.shape

	width = gt_shape[0]
	height = gt_shape[1]
	width2 = disp_shape[0]
	height2 = disp_shape[1]
	scale = width / width2

	if ((scale != 1 and scale != 2 and scale != 4)
	or ((scale * width2) != width)
	or ((scale * height2) != height)):
		print("	  disp size = ", width2, " x ", height2)
		print("GT disp size =", width, " x ", height)
		raise ValueError("GT disp size must be exactly 1, 2, or 4 * disp size")

	n = 0
	bad = 0
	invalid = 0
	serr = 0.0

	for y in range(height):
		for x in range(width):
			gt = gtdisp[x, y]
			
			d = float(scale * disparity[int(x/scale), int(y/scale)])
			valid = (d != 0)
			if not valid:
				invalid += 1

			err = abs(d-gt)
			n += 1
			if (valid):
				serr += err
				if (err > badthresh):
					bad += 1

	badpercent =  100.0*bad/n
	invalidpercent =  100.0*invalid/n
	totalbadpercent =  100.0*(bad+invalid)/n
	avgErr = serr / (n - invalid)

	print("Rated pixels: ", 100.0*n/(width*height), 
		  "\nBad percentage: ", badpercent, 
		  "\nInvalid percentage: ", invalidpercent, 
		  "\nTotal bad percentage: ", totalbadpercent, 
		  "\nAverage error: ", avgErr, '\n')

base = os.path.abspath(os.path.dirname(__file__))
if os.name == 'nt':
	base_new = base.replace('\\f_aux', '')
else:
	base_new = base.replace('/f_aux', '')

# Define vectors for images and their respective paths
images = ['disparidade.npy', 'gt_disparidade.npy']
data = [os.path.join(base_new, 'data', 'Middlebury', 'Jadeplant-perfect'),
		os.path.join(base_new, 'data', 'Middlebury', 'Playtable-perfect')]
name = ['Jadeplant', 'Playtable']

for i in range(len(name)):
	disparity = np.load(os.path.join(data[i], images[0]))*100
	gtdisp = np.load(os.path.join(data[i], images[1]))*100

	# Normalize data to show as an image
	img_d = cv.normalize(disparity, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
	img_gt = cv.normalize(gtdisp, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

	cv.namedWindow('Disparity map', cv.WINDOW_NORMAL)
	cv.resizeWindow('Disparity map', (439, 331))
	cv.imshow('Disparity map', img_d)

	cv.namedWindow('GroundTruth', cv.WINDOW_NORMAL)
	cv.resizeWindow('GroundTruth', (439, 331))
	cv.imshow('GroundTruth', img_gt)

	cv.waitKey(0)
	cv.destroyAllWindows()

	badthresh = 2

	print('Calculating error to ' + name[i] + ' results...\n', flush=True)
	evaldisp(disparity, gtdisp, badthresh)