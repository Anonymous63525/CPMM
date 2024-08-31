import gzip
import pdb
from collections import defaultdict
from datetime import datetime
import os
import copy
import json
import requests
from bs4 import BeautifulSoup
import requests
from PIL import Image
from io import BytesIO

false = False
true = True


def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)


def meta_to_5core(core5_dataset='original_data/reviews_Beauty_5', meta_DATASET='original_data/meta_Beauty',
                  save_path='image_text/beauty .json'):
    dataname = core5_dataset + '.json.gz'
    meta_dataname = meta_DATASET + '.json.gz'

    countU = defaultdict(lambda: 0)
    countP = defaultdict(lambda: 0)
    for one_interaction in parse(dataname):
        rev = one_interaction['reviewerID']
        asin = one_interaction['asin']
        countU[rev] += 1
        countP[asin] += 1

    meta_dict = dict()
    for item_meta in parse(meta_dataname):
        meta_dict[item_meta['asin']] = item_meta

    itemmap = dict()
    itemnum = 1
    num = 1
    core5_item_dict = {}

    for one_interaction in parse(dataname):
        rev = one_interaction['reviewerID']
        asin = one_interaction['asin']
        if countU[rev] < 5 or countP[asin] < 5:
            continue

        if asin in itemmap:
            continue
        else:
            num += 1
            itemid = itemnum
            itemmap[asin] = itemid
            itemnum += 1
            core5_item_dict[asin] = meta_dict[asin]

    if not os.path.exists(save_path.split('/')[0]):
        os.makedirs(save_path.split('/')[0])

    with open(save_path, mode='w', encoding='utf-8') as f:
        json.dump(core5_item_dict, f, ensure_ascii=True)


def extract_text(item_path='image_text/beauty.json', save_path='image_text/beauty_text.json', text='title'):
    with open(item_path, 'r') as json_file:
        data = json.load(json_file)

    count = 0
    no_text = []
    item_text = {}
    for _, one_interaction in data.items():

        if text not in one_interaction:
            no_text.append(one_interaction)
            continue

        count += 1
        asin = one_interaction['asin']
        item_text[asin] = one_interaction[text]


    if not os.path.exists(save_path.split('/')[0]):
        os.makedirs(save_path.split('/')[0])
    with open(save_path, mode='w', encoding='utf-8') as f:
        json.dump(item_text, f, ensure_ascii=True)



if __name__ == '__main__':
    filename = 'Beauty'
    dataname = 'Beauty'

    meta_to_5core(core5_dataset=f'original_data/reviews_{filename}_5', meta_DATASET=f'original_data/meta_{filename}',
                   save_path=f'image_text/{dataname}.json')

    extract_text(item_path=f'image_text/{dataname}.json', save_path=f'image_text/{dataname}_text.json', text='title')

