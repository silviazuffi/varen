# VAREN
This repository contains the VAREN horse model describen in the paper : _VAREN: Very Accurate and Realistic Equine Network_, by Silvia Zuffi, Ylva Mellbin, Ci Li, Markus Hoeschle, Hedvig Kjellström, Senya Polikovsky, Elin Hernlund, and Michael J. Black, CVPR 2024. 

![teaser](./images/teaser_larger1.png)

## Installation
Follow the instructions to install [pytorch3d](https://github.com/facebookresearch/pytorch3d/tree/main)
Create a conda environment and install the required packages.
```
conda activate pytorch3d
conda install pip
pip install absl-py
```
Clone the repository and then download the pre-trained [network](https://).
Place the network in the folder:
```
varen/code/cachedir/snapshots/varen/
```
Download the [horse smal model](https://) and place it in the folder:
```
varen/model/
```
If you want to run the training code you need the [dataset](https://).
Place the scans and the registrations in the folders:
```
varen/scans/decimated_clean/
varen/registrations/
```
If you want to compute the errors on the testset you need the [testset](https://).
Place the testset in the folder
```
varen/data/testset_outside_shape_space
varen/data/testset_inside_shape_space
```



## Running the code
To retrain the model:
```
./train.sh
```
To compute the errors on the testset (by default on the outside shape space data):
```
./predict.sh
```



## Citation

If you found the model or any of the pieces of code useful in this repo, please cite the paper:

```
@inproceedings{Zuffi:CVPR:2024,  
  title = {{VAREN}: Very Accurate and Realistic Equine Network},  
  author = {Zuffi, Silvia and Mellbin, Ylva and Li, Ci and Hoeschle, Markus and Kjellström, Hedvig and Polikovsky, Senya  and Hernlund, Elin and Black, Michael J.},  
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},  
  pages = {},
  month = Jun,
  year = 2024
}










