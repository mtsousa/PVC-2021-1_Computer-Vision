from pathlib import Path
import numpy as np
import csv
import re
import cv2
import os.path
import matplotlib.pyplot as plt

def read_calib(calib_file_path):
    with open(calib_file_path, 'r') as calib_file:
        calib = {}
        csv_reader = csv.reader(calib_file, delimiter='=')
        for attr, value in csv_reader:
            calib.setdefault(attr, value)

    return calib

def read_pfm(pfm_file_path):
    with open(pfm_file_path, 'rb') as pfm_file:
        header = pfm_file.readline().decode().rstrip()
        channels = 3 if header == 'PF' else 1

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', pfm_file.readline().decode('utf-8'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception("Malformed PFM header.")

        scale = float(pfm_file.readline().decode().rstrip())
        if scale < 0:
            endian = '<' # littel endian
            scale = -scale
        else:
            endian = '>' # big endian

        dispariy = np.fromfile(pfm_file, endian + 'f')
    #

    # Eu mexi daqui
    img = np.reshape(dispariy, newshape=(height, width, channels))
    img[img==np.inf] = 0
    img = np.flipud(img)
    #

    
    groundtruth = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    show(groundtruth, "disparity")
    cv2.imwrite('gt_disparity.png', groundtruth)

    plt.imshow(groundtruth, cmap='jet')
    plt.colorbar()
    plt.savefig('color_gt_disparity.png')
    plt.show()
    # Até aqui...
    # O resto veio daqui: https://blog.csdn.net/weixin_44899143/article/details/89186891

    return dispariy, [(height, width, channels), scale]


def create_depth_map(pfm_file_path, calib=None):

    dispariy, [shape,scale] = read_pfm(pfm_file_path)

    if calib is None:
        raise Exception("Loss calibration information.")
    else:
        fx = float(calib['cam0'].split(' ')[0].lstrip('['))
        base_line = float(calib['baseline'])
        doffs = float(calib['doffs'])

		# scale factor is used here
        depth_map = fx*base_line / (dispariy / scale + doffs)
        depth_map = np.reshape(depth_map, newshape=shape)
        depth_map = np.flipud(depth_map).astype('uint8')

        return depth_map

def show(img, win_name='image'):
    if img is None:
        raise Exception("Can't display an empty image.")
    else:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.imshow(win_name, img)
        cv2.waitKey()
        cv2.destroyWindow(win_name)


def main():
    base = os.path.abspath(os.path.dirname(__file__))
    base_new = os.path.join(base, 'data', 'Middlebury', 'Jadeplant-perfect')

    pfm_file_dir = Path(base_new)
    calib_file_path = pfm_file_dir.joinpath('calib.txt')
    disp_left = pfm_file_dir.joinpath('disp0.pfm')
	
    # calibration information
    calib = read_calib(calib_file_path)
	# create depth map
    depth_map_left = create_depth_map(disp_left, calib)

    show(depth_map_left, "depth_map")

if __name__ == '__main__':
    main()
#————————————————
#版权声明：本文为CSDN博主「2h4dl」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
#原文链接：https://blog.csdn.net/weixin_44899143/article/details/89186891