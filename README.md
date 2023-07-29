# Knowledge Combination to Learn Rotated Detection Without Rotated Annotation
This is the implementation of the paper: Knowledge Combination to Learn Rotated Detection Without Rotated Annotation. The paper is accepted by CVPR2023. We build KCR with mmcv, mmdet and mmrotate. 
Paper Link: https://openaccess.thecvf.com/content/CVPR2023/papers/Zhu_Knowledge_Combination_To_Learn_Rotated_Detection_Without_Rotated_Annotation_CVPR_2023_paper.pdf

## Getting Started
Install the main packages:
 - torch==1.9.1
 - mmcv==1.5.3
Then use the following command to setup KCR:
```
bash setup
```
For the dataset setup, please follow coco. We convert all the dataset to the form of coco instance detection. 

## Training
```
cd kcrtools
python train_kcr_2d.py
```
Please provide the rotated dataset such as coco and axis-aligned dataset in the config file. 

# Inference
```
cd kcrtools
python test_rotated_2d.py
```
