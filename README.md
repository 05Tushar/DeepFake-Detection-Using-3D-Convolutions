# DeepFake Detection using 3D Conv-nets

This code uses 3D Convolutional Neural Networks to detect Deepfake Videos. 3D Conv-nets are rich in both spatial and temporal information, hence it makes sense to use them to detect DeepFake videos. 
The code for 3D Conv-net is taken from 
[Kensho Hara 3D-ResNets-PyTorch](https://github.com/kenshohara/3D-ResNets-PyTorch)

## Dataset 

Dataset for training can be found on [Kaggle DFDC Challenge](https://www.kaggle.com/c/deepfake-detection-challenge/data)

## Working
First, individual frames are extracted from both the test and train videos. The 3-D Conv-net takes as input frames of size 112 x 112 x 16 x 3[RGB Channels]. Faces cropped to size 112 x 112 are extracted from individual frames using OpenCV’s face detection neural network.  The first frame of these 16 frames is chosen randomly from the whole sequence of frames.

## Requirements
We have used Conda to download all the necessary requirements 

## Preparation
To convert the videos to individual frames use 

```bash
python -m util_scripts.generate_video_jpgs avi_video_dir_path jpg_video_dir_path faceforensics
```

To create the JSON file necessary for training use 

```bash
python -m util_scripts.faceforensics_json annotation_dir_path jpg_video_dir_path dst_json_path
```
The annotation directory contains the metadata.json file that comes along with the DFDC data set.

Download the pretrained dataset from [kenshohara/3D-ResNets-PyTorch#pre-trained-models](https://github.com/kenshohara/3D-ResNets-PyTorch#pre-trained-models)

## Training
To train the network use 
```bash
python main.py --root_path ~/data --video_path faceforensics_videos/jpg --annotation_path faceforensics.json \ 
--result_path results --dataset faceforensics --n_classes 2 --n_pretrain_classes 1039 \
--pretrain_path models/resnet-20-kinetics.pth --ft_begin_module fc \
--model resnet --model_depth 200 --batch_size 128 --n_threads 4 --checkpoint 5
```
