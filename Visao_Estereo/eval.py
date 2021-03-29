import cv2 as cv
import numpy as np

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
		raise ValueError("GT disp size must be exactly 1, 2, or 4 * disp size");

	n = 0
	bad = 0
	invalid = 0
	serr = 0.0
	print(disparity)
	print(gtdisp)
	for y in range(height):
		for x in range(width):
			gt = gtdisp[x, y]

			if (gt == np.inf):
				continue

			d = float(scale * disparity[int(x/scale), int(y/scale)])
			valid = (d != np.inf)

			if (valid):
				maxd = scale * maxdisp
				d = max(0, min(maxd, d))
				
			if (valid and rounddisp):
				d = round(d)

			err = abs(d-gt)
			n += 1

			if (valid):
				serr += err
				if (err > badthresh):
					bad += 1
					pass
				pass
			pass
		pass

	badpercent =  100.0*bad/n
	invalidpercent =  100.0*invalid/n
	totalbadpercent =  100.0*(bad+invalid)/n
	avgErr = serr / (n - invalid)

	print(100.0*n/(width*height), "Bad percentage: ", badpercent, "Invalid percentage: ", invalidpercent, "Total bad percentage: ", totalbadpercent, "Average error: ", avgErr)

	pass

disparity = cv.imread('disparidade.pgm', cv.IMREAD_GRAYSCALE)
gtdisp = cv.imread('disp0-n.pgm', cv.IMREAD_GRAYSCALE)

cv.namedWindow('Disparidade calculada', cv.WINDOW_NORMAL)
cv.resizeWindow('Disparidade calculada', (439, 331))
cv.imshow('Disparidade calculada', disparity)

cv.namedWindow('GroundTruth disponibilizado', cv.WINDOW_NORMAL)
cv.resizeWindow('GroundTruth disponibilizado', (439, 331))
cv.imshow('GroundTruth disponibilizado', gtdisp)

cv.waitKey(0)

badthresh = 2
maxdisp = 1988 / 3
rounddisp = 0

evaldisp(disparity, gtdisp, badthresh, maxdisp, rounddisp)
