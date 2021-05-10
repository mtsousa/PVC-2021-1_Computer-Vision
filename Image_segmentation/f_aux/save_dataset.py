# -*- coding: utf-8 -*-
# Autores: Matheus Teixeira de Sousa (teixeira.sousa@aluno.unb.br)
#          João Luiz Machado Júnior (180137158@aluno.unb.br)
# Disciplina: Princípios de Visão Computacional - turma A

import time
import concurrent.futures
import argparse
import urllib.request
import shutil
import os

IMAGES_DIR = 'images/COCO_images'

cont = 0
def download_image(img_url):
    global cont
    img_name = img_url[len(img_url)-16:len(img_url)]

    with urllib.request.urlopen(img_url) as resp, open(IMAGES_DIR + '/' + img_name, 'wb') as out:
        shutil.copyfileobj(resp, out)
        cont += 1
        if cont%1000 == 0:
            print('Images downloaded:', cont)

# Parse command line arguments
parser = argparse.ArgumentParser(
	description='Download COCO dataset.')
parser.add_argument("command",
					metavar="<command>",
					help="'train' or 'val' on MS COCO")

data = ['filtered_coco/url_train.txt', 'filtered_coco/url_val.txt',
        'filtered_coco/annotations/instances_train2017.json',
        'filtered_coco/annotations/instances_val2017.json']

args = parser.parse_args()
if args.command == 'train':
	dir_data = data[0]
elif args.command == 'val':
	dir_data = data[1]

print('save_dataset mode:', args.command)
print('directory of url data:', dir_data)

TRAIN_URL = 'https://github.com/mtsousa/Visao_Computacional/releases/download/v1_train/instances_train2017.json'
VAL_URL = 'https://github.com/mtsousa/Visao_Computacional/releases/download/v1_val/instances_val2017.json'

# Download annotations if does not exists
if not (os.path.exists(data[2]) or os.path.exists(data[3])):
    print('Downloading annotations...')
    with urllib.request.urlopen(TRAIN_URL) as resp, open(data[2], 'wb') as out:
        shutil.copyfileobj(resp, out)
    with urllib.request.urlopen(VAL_URL) as resp, open(data[3], 'wb') as out:
        shutil.copyfileobj(resp, out)

# Read urls on url file
with open(dir_data , 'r') as url_list:
    img_urls = [str(line[2:len(line)-3]) for line in url_list.readlines()]

t1 = time.perf_counter()

# For each url, download the image
with concurrent.futures.ThreadPoolExecutor() as executor:
    print('Images to download:', len(img_urls))
    for url in img_urls:
        executor.submit(download_image, url)

t2 = time.perf_counter()

print(f'Finished in {t2-t1} seconds')
print('Images downloaded are in images/COCO_images/')