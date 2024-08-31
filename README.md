# CPMM
This is the implementation for the paper: One Multimodal Plugin Enhancing All: CLIP-based Pre-training
Multimodal Item Representations for Recommendation System

## Code introduction
The code is implemented based on [S3-Rec](https://github.com/RUCAIBox/CIKM2020-S3Rec) and [RecBole](https://github.com/RUCAIBox/RecBole)

### Datasets
We used the Amazon standard dataset and stored the feature vectors extracted from CLIP [here](https://drive.google.com/file/d/1_oUco-jcc1MxrWiAK8OYbJNIxKsfkn4x/view?usp=sharing).


### Pretrain
First, use CPMM to pretrain representations.

#### SR
```angular2html
python CPMM_main.py --lr=1e-4 --epoch=1000 --batchsize=128 --std=0.002 --dataname='Beauty' --recommendation_task='SR'
```
#### CF
```angular2html
python CPMM_main.py --lr=1e-4 --epoch=1000 --batchsize=128 --std=0.002 --dataname='Beauty' --recommendation_task='CF'
```
#### CTR
```angular2html
python CPMM_main.py --lr=1e-4 --epoch=1000 --batchsize=128 --std=0.004 --dataname='Beauty' --recommendation_task='CTR'
```

### Downstream tasks

#### SR
```angular2html
cd SR
python main_sr.py  --dataname='Beauty' --pretraining='True'
```

#### CF
```angular2html
cd CF
python main_cf.py  --dataname='Beauty' --pretraining='True'
```

#### CTR
```angular2html
cd CTR
python main_ctr.py --dataname='Beauty' --model_name='xDeepFM' --pretraining=True --batchsize=1024 --lr=0.0001 --dropout_prob=0.1 --weight_decay=0.0001
```