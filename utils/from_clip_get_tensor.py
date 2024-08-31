import pdb
import os
from PIL import Image
import pandas as pd
import gzip
import json
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch
import clip
from tqdm import tqdm
from PIL import Image
def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)


def get_itemmap(DATASET='data/original_data/reviews_Beauty_5'):
    false = False
    true = True
    dataname = DATASET + '.json.gz'
    itemmap = dict()
    itemnum = 1
    for one_interaction in parse(dataname):
        asin = one_interaction['asin']

        if asin in itemmap:
            itemid = itemmap[asin]
        else:
            itemid = itemnum
            # 给商品编号
            itemmap[asin] = itemid
            itemnum += 1
    return itemmap


def read_text_image(itemmap, text_path='image_text/beauty_text.json', image_path="image_text/beauty_image"):
    text = {}
    image = {}
    parent_folder = image_path
    subfolders = [f.split('.')[0] for f in os.listdir(parent_folder)]
    with open(text_path, 'r') as json_file:
        text_data = json.load(json_file)

    for map_k, map_v in itemmap.items():
        if map_k in text_data and map_k in subfolders:
            text[map_v] = text_data[map_k]

            with Image.open(os.path.join(image_path, f'{map_k}.png')) as image_data:
                image_tensor = torch.tensor(np.array(image_data))
                image[map_v] = image_tensor

    return text, image

dataname = 'Tools'
if dataname == 'Beauty':
    ori_path = 'reviews_Beauty_5'
    name = 'beauty'
elif dataname == 'Sport':
    ori_path = 'reviews_Sports_and_Outdoors_5'
    name = 'sport'
elif dataname == 'Tools':
    ori_path = 'reviews_Tools_and_Home_Improvement_5'
    name = 'tools'
elif dataname == 'Toys':
    ori_path = 'reviews_Toys_and_Games_5'
    name = 'toys'


itemmap = get_itemmap(DATASET=f'original_data/{ori_path}')


text_dict, image_dict = read_text_image(itemmap, text_path=f'image_text/{name}_text.json',
                                        image_path=f"image_text/{name}_image")


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)



text_features_list = torch.Tensor().to(device)
image_features_list = torch.Tensor().to(device)

item_num = 1
for k, v in tqdm(itemmap.items()):
    if (v in text_dict) and (v in image_dict):

        image_ = preprocess(Image.open(f"image_text/{name}_image/{k}.png")).unsqueeze(0).to(device)
        text = clip.tokenize(text_dict[v], truncate=True).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_)
            text_features = model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            text_features_list = torch.concat((text_features_list, text_features), dim=0)
            image_features_list = torch.concat((image_features_list, image_features), dim=0)
    else:
        text_features_list = torch.concat((text_features_list, torch.normal(mean=0.0, std=0.02, size=(1, text_features_list.shape[-1])).to(device)), dim=0)
        image_features_list = torch.concat((image_features_list, torch.normal(mean=0.0, std=0.02, size=(1, image_features_list.shape[-1])).to(device)), dim=0)


# 由clip获取的image和text的embedding
torch.save(text_features_list, f'clip_text_features_{dataname}.pt')
torch.save(image_features_list, f'clip_image_features_{dataname}.pt')

