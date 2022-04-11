# MST++
This is the implementation of our proposed solution "MST++: Multi-stage Spectral-wise Transformer for Efficient Spectral Reconstruction". Our MST++ is mainly based on our work [MST](https://github.com/caiyuanhao1998/MST), which is accepted by CVPR 2022.


![Illustration of MST](/figure/MST.png)


This repo includes:  

- Specification of dependencies.
- Testing code (both development and challenge result).
- Training code.
- Pre-trained models.


## 1. Create Envirement:

- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))

- [PyTorch >= 1.3](https://pytorch.org/) 

- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

- Python packages:

  ```shell
  pip install -r requirements.txt
  ```


## 2. Reproduce the development result:

(1)  Download the pretrained model zoo from [Google Drive](https://drive.google.com/drive/folders/1GzsNbd-XC8UZEq5V4JaisEyCSKVihCQG?usp=sharing) and place them to ' /source_code/test_develop_code/model_zoo/'. 

(2)  Download the validation RGB images from [Google Drive](https://drive.google.com/file/d/19vBR_8Il1qcaEZsK42aGfvg5lCuvLh1A/view)  and place them to ' /source_code/test_develop_code/Valid_RGB/'. 

(3)  Test our models on the validation RGB images. The results will be saved in '/MST-plus-plus/test_develop_code/results_model_ensemble/submission/submission.zip' in the zip format. 

```shell
cd /MST-plus-plus/test_develop_code/
python test.py --pretrained_model_path ./model_zoo/MstPlus_1stg_ps128_s8_norm.pth --outf ./exp/mst_plus_plus/
```



## 3. Reproduce the challenge result:

(1)  Download the pretrained model zoo from [Google Drive](https://drive.google.com/drive/folders/1pAzS3YY8-Av49i-uoF7GLzodnt1qYReL?usp=sharing) and place them to ' /MST-plus-plus/test_challenge_code/model_zoo/'. 

(2)  Download the testing RGB images from [Google Drive](https://drive.google.com/file/d/1A5309Gk7kNFI-ORyADueiPOCMQNTA7r5/view)  and place them to ' /MST-plus-plus/test_challenge_code/Test_RGB/'. 

(3)  Test our models on the testing RGB images. The results will be saved in '/MST-plus-plus/test_challenge_code/results_model_ensemble/submission/submission.zip' in the zip format. 

```shell
cd /MST-plus-plus/test_challenge_code/
python test.py --pretrained_model_path ./model_zoo/MST_plus_1stg_lr4e-4_s8_norm_DevValid.pth --outf ./exp/mst_plus_plus/
```



## 4. Training

(1)  Data preparation:

- Download [training spectral images](https://drive.google.com/file/d/1FQBfDd248dCKClR-BpX5V2drSbeyhKcq/view), [training RGB images](https://drive.google.com/file/d/1A4GUXhVc5k5d_79gNvokEtVPG290qVkd/view),  [validation spectral images](https://drive.google.com/file/d/12QY8LHab3gzljZc3V6UyHgBee48wh9un/view), [validation RGB images](https://drive.google.com/file/d/19vBR_8Il1qcaEZsK42aGfvg5lCuvLh1A/view), and [testing RGB images](https://drive.google.com/file/d/1A5309Gk7kNFI-ORyADueiPOCMQNTA7r5/view) from the [competition website](https://codalab.lisn.upsaclay.fr/competitions/721#participate-get_data).

- Place the training spectral images and validation spectral images to "/MST-plus-plus/train_code/ARAD_1K/Train_Spec/".

- Place the training RGB images and validation RGB images to "/MST-plus-plus/train_code/ARAD_1K/Train_RGB/".

- Then the code are collected as the following form:

```
  |--MST-plus-plus
  |	|--test_challenge_code
  |	|--test_develop_code
  |	|--train_code  
  |	|	|--ARAD_1K 
  |	|	|	|--Train_Spec
  |	|       |       |	|--ARAD_1K_0001.mat
  |	|       |       |	|--ARAD_1K_0001.mat
  |	|       |       |	： 
  |	|       |       |	|--ARAD_1K_0950.mat
  |	|       |    	|--Train_RGB
  |	|       |    	|	|--ARAD_1K_0001.jpg
  |	|       |       |	|--ARAD_1K_0001.jpg
  |	|       |       |	： 
  |	|       |       |	|--ARAD_1K_0950.jpg
```


(2)  To train a single model, run

```shell
cd /MST-plus-plus/train_code/
python main.py --method mst_plus_1stg --gpu_id 0 --batch_size 20 --init_lr 4e-4 --outf ./exp/ --data_root ./ARAD_1K/  --patch_size 128 --stride 8 -norm
```




## Citation
```
@inproceedings{mst,
	title={Mask-guided Spectral-wise Transformer for Efficient Hyperspectral Image Reconstruction},
	author={Yuanhao Cai and Jing Lin and Xiaowan Hu and Haoqian Wang and Xin Yuan and Yulun Zhang and Radu Timofte and Luc Van Gool},
	booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	year={2022}
}
```
