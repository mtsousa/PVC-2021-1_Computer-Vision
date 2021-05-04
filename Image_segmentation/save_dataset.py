# -*- coding: utf-8 -*-
import requests
import time
import concurrent.futures
import argparse

cont = 0
def download_image(img_url):
    global cont
    img_bytes = requests.get(img_url).content
    img_name = img_url[len(img_url)-16:len(img_url)]
    # Salva a imagem dentro do ambiente de execução
    with open('/content/images/' + img_name, 'wb') as img_file:
        img_file.write(img_bytes)
        cont += 1
        if cont%1000 == 0:
            # Essa linha pode resultar em IOPub data rate exceeded
            # Caso dê o erro, basta comentar o print
            print('Images downloaded:', cont)

data = ['/content/drive/MyDrive/Image_segmentation/filtered_coco/url_train.txt', 
        '/content/drive/MyDrive/Image_segmentation/filtered_coco/url_val.txt']

# Parse command line arguments
parser = argparse.ArgumentParser(
	description='Download COCO dataset.')
parser.add_argument("command",
					metavar="<command>",
					help="'train' or 'val' on MS COCO")

args = parser.parse_args()
if args.command == 'train':
	dir_data = data[0]
elif args.command == 'val':
	dir_data = data[1]

print('save_dataset mode:', args.command)
print('directory of url data:', dir_data)

# Endereço do arquivo de url das imagens de treino
with open(dir_data , 'r') as url_list:
    img_urls = [str(line[2:len(line)-3]) for line in url_list.readlines()]

t1 = time.perf_counter()

with concurrent.futures.ThreadPoolExecutor() as executor:
    print('Images to download:', len(img_urls))
    for url in img_urls:
        executor.submit(download_image, url)

t2 = time.perf_counter()

print(f'Finished in {t2-t1} seconds')
print('Images downloaded are in /content/images/')