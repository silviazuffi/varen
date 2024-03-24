# VAREN
This repository contains the VAREN horse model describen in the paper : _VAREN: Very Accurate and Realistic Equine Network_, by Silvia Zuffi, ...., and Michael J. Black, CVPR 2024. 
![teaser](./images/VAREN.png)

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
```



## Running the code
To retrain the model:
```
./train.sh
```
To compute the errors on the testset:
```
./predict.sh
```



## Citation

If you found any of the pieces of code useful in this repo, please cite the following papers:

```
@inproceedings{Zuffi:CVPR:2024,  
  title = {VAREN: Very Accurate and Realistic Equine Network},  
  author = {Zuffi, Silvia and .... and Black, Michael J.},  
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},  
  pages = {},  
  publisher = {IEEE Computer Society},  
  year = {2024}. 
}










