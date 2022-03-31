# MST
This is the pytorch implementation of our proposed solution "MST++: Multi-stage Spectral-wise Transformer for Efficient Spectral Reconstruction".


Code and models are coming soon.

![Illustration of MST](/figure/MST.png)


This repo includes:  

- Specification of dependencies.
- Testing code (both development and challenge result).
- Training code.
- Pre-trained models.
- README file.

This repo can reproduce the development and challenge result of our team .
All the source code and pre-trained models will be released to the public for further research.


#### 1. Create Envirement:

------

- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))

- [PyTorch >= 1.3](https://pytorch.org/) 

- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

- Python packages:

  ```shell
  pip install -r requirements.txt
  ```

#### 2. Reproduce the development result:

(1)  Download the pretrained model zoo from [Google Drive](https://drive.google.com/drive/folders/1pZ7wcFXU8Y9HFvViRA0QMvJIkvNXfhLC?usp=sharing) and place them to ' /source_code/test_develop_code/model_zoo/'. 

(2)  Download the validation RGB images from [Google Drive](https://drive.google.com/file/d/19vBR_8Il1qcaEZsK42aGfvg5lCuvLh1A/view)  and place them to ' /source_code/test_develop_code/Valid_RGB/'. 

(3)  Test our models on the validation RGB images. The results will be saved in '/MST-plus-plus/test_develop_code/results_model_ensemble/submission/submission.zip' in the zip format. 

```shell
cd /MST-plus-plus/test_develop_code/
python test.py --pretrained_model_path ./model_zoo/MstPlus_1stg_ps128_s8_norm.pth --outf ./exp/mst_plus_plus/
```

#### 3. Reproduce the challenge result:

(1)  Download the pretrained model zoo from [Google Drive](https://drive.google.com/drive/folders/17RbgxylNTZo73Lgx0bcMcd69hg_NE2EJ?usp=sharing) and place them to ' /MST-plus-plus/test_challenge_code/model_zoo/'. 

(2)  Download the testing RGB images from [Google Drive](https://drive.google.com/file/d/1A5309Gk7kNFI-ORyADueiPOCMQNTA7r5/view)  and place them to ' /MST-plus-plus/test_challenge_code/Test_RGB/'. 

(3)  Test our models on the testing RGB images. The results will be saved in '/MST-plus-plus/test_challenge_code/results_model_ensemble/submission/submission.zip' in the zip format. 

```shell
cd /MST-plus-plus/test_challenge_code/
python test.py --pretrained_model_path ./model_zoo/MST_plus_1stg_lr4e-4_s8_norm_DevValid.pth --outf ./exp/mst_plus_plus/
```

#### 4. Training

(1)  Data preparation:

- Download training spectral images, training RGB images,  validation spectral images, validation RGB images from the [competition website](https://codalab.lisn.upsaclay.fr/competitions/721#participate-get_data).

- Place the training spectral images and validation spectral images to "/MST-plus-plus/train_code/ARAD_1K/Train_Spec/".

- Place the training RGB images and validation RGB images to "/MST-plus-plus/train_code/ARAD_1K/Train_RGB/".

- Then the code are collected as the following form:

  	|--MST-plus-plus
  		|--test_challenge_code
  		|--test_develop_code
  	    |--train_code  
  	        |--ARAD_1K 
  	            |--Train_Spec
  	                |--ARAD_1K_0001.mat
  	                |--ARAD_1K_0001.mat
  	                ： 
  	                |--ARAD_1K_0950.mat
  	            |--Train_RGB
  	            	|--ARAD_1K_0001.jpg
  	                |--ARAD_1K_0001.jpg
  	                ： 
  	                |--ARAD_1K_0950.jpg


(2)  To train a single model, run

```shell
cd /MST-plus-plus/train_code/
python main.py --method mst_plus_1stg --gpu_id 0 --batch_size 20 --init_lr 4e-4 --outf ./exp/ --data_root ./ARAD_1K/  --patch_size 128 --stride 8 -norm
```

#### 5. This repo is mainly based on MST and AWAN.  In our experiments, we use the following repos:

(1)  MST: https://github.com/caiyuanhao1998/MST

(2)  AWAN: https://github.com/Deep-imagelab/AWAN

(3)  MIRNet:  https://github.com/swz30/MIRNet

(4)  MPRNet: https://github.com/swz30/MPRNet

(5)  Restormer: https://github.com/swz30/Restormer

We thank these repos and have cited these works in our manuscript.

#### Citation
```
@inproceedings{mst,
	title={Mask-guided Spectral-wise Transformer for Efficient Hyperspectral Image Reconstruction},
	author={Yuanhao Cai and Jing Lin and Xiaowan Hu and Haoqian Wang and Xin Yuan and Yulun Zhang and Radu Timofte and Luc Van Gool},
	booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	year={2022}
}
```
