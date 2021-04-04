import cv2 as cv
import numpy as np
import os.path

base = os.path.abspath(os.path.dirname(__file__))
if os.name == 'nt':
    base_new = base.replace('\\f_aux', '')
else:
    base_new = base.replace('/f_aux', '')

# Define os vetores das imagens e dos caminhos para as imagens
images = ['im0.png', 'im1.png', 'rectifiedL.jpg', 'rectifiedR.jpg']
# images = ['im0.png', 'im1.png', 'rectifiedR.jpg', 'rectifiedL.jpg']
data = [os.path.join(base_new, 'data', 'Middlebury', 'Jadeplant-perfect'),
        os.path.join(base_new, 'data', 'Middlebury', 'Playtable-perfect'),
        os.path.join(base_new, 'data', 'FurukawaPonce')]

imgL = cv.imread(os.path.join(data[2], images[2]), cv.IMREAD_GRAYSCALE)
imgR = cv.imread(os.path.join(data[2], images[3]), cv.IMREAD_GRAYSCALE)
#stereo = cv.StereoBM_create(numDisparities=16*29, blockSize=15)

win_size = 15**2
block = 5
min_disp = -8
max_disp = 16*4
num_disp = max_disp - min_disp  # Needs to be divisible by 16
left_matcher = cv.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=max_disp,
    blockSize=block,
    uniquenessRatio=10,
    speckleWindowSize=10,
    speckleRange=2,
    disp12MaxDiff=-1,
    P1=8*3*win_size,
    P2=32*3*win_size,
    preFilterCap=63,
    mode=cv.STEREO_SGBM_MODE_SGBM_3WAY
)

stereo = left_matcher
right_matcher = cv.ximgproc.createRightMatcher(left_matcher)

# FILTER Parameters
lmbda = 80000
sigma = 1.3

wls_filter = cv.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
wls_filter.setLambda(lmbda)

wls_filter.setSigmaColor(sigma)
displ = left_matcher.compute(imgL, imgR) #.astype(np.float32)/16
dispr = right_matcher.compute(imgR, imgL) #.astype(np.float32)/16
displ = np.int16(displ)
dispr = np.int16(dispr)
filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!

filteredImg = cv.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv.NORM_MINMAX)
filteredImg = np.uint8(filteredImg)


disparity = stereo.compute(imgL,imgR)
norm_image = cv.normalize(disparity, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

cv.namedWindow('image1', cv.WINDOW_NORMAL)
cv.resizeWindow('image1', (439, 331))
cv.imshow('image1', norm_image)

# filteredImg = cv.GaussianBlur(filteredImg, (5,5), 5, None, 5)

cv.namedWindow('image2', cv.WINDOW_NORMAL)
cv.resizeWindow('image2', (439, 331))
cv.imshow('image2', filteredImg)
cv.waitKey(0)