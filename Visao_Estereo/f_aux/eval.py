import cv2 as cv
import numpy as np
import os.path
import matplotlib.pyplot as plt

def evaldisp(disparity, gtdisp, badthresh, maxdisp, rounddisp):

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
	#print(disparity)
	#print(gtdisp)
	if np.any(gtdisp[gtdisp == 0]):
		print('Tem zero')
	for y in range(height):
		for x in range(width):
			gt = gtdisp[x, y]

			if (gt == np.inf):
				print('Eh infinito')

			d = float(scale * disparity[int(x/scale), int(y/scale)])
			if d == np.inf:
				print('Eh infinito')
			valid = (d != 0)#np.inf)
			if not valid:
				invalid += 1

			if d > maxdisp:
				print('Eh maior')

			if (valid):
				maxd = scale * maxdisp
				d = max(0, min(maxd, d))
				
			if (valid and rounddisp):
				print('Entrei')
				d = round(d)

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

# Define os vetores das imagens e dos caminhos para as imagens
images = ['disparidade.npy', 'gt_disparidade.npy']
data = [os.path.join(base_new, 'data', 'Middlebury', 'Jadeplant-perfect'),
		os.path.join(base_new, 'data', 'Middlebury', 'Playtable-perfect')]
name = ['Jadeplant', 'Playtable']

for i in range(len(name)):
	disparity = np.load(os.path.join(data[i], images[0]))*100
	gtdisp = np.load(os.path.join(data[i], images[1]))*100

	#mask_resize = cv.resize(mask, (disparity.shape[1], disparity.shape[0]), interpolation = cv.INTER_LANCZOS4)

	cv.namedWindow('Disparidade calculada', cv.WINDOW_NORMAL)
	cv.resizeWindow('Disparidade calculada', (439, 331))
	cv.imshow('Disparidade calculada', disparity)

	cv.namedWindow('GroundTruth disponibilizado', cv.WINDOW_NORMAL)
	cv.resizeWindow('GroundTruth disponibilizado', (439, 331))
	cv.imshow('GroundTruth disponibilizado', gtdisp)

	cv.waitKey(0)
	cv.destroyAllWindows()

	badthresh = 2
	maxdisp = 1988 / 3
	rounddisp = 0

	print('Calculating error to ' + name[i] + ' results...\n', flush=True)
	evaldisp(disparity, gtdisp, badthresh, maxdisp, rounddisp)