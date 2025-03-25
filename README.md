# Compensation Compress

This repository is the Pytorch implementation of the paper "Deep Lossy Compression with more Precise Reconstruction Estimation"

**Note:** We highly appreciate the contribution of [Minnen's ICIP2020 paper](https://ieeexplore.ieee.org/document/9190935/), they successfully integrate the Straight-through Estimator into existing LIC paradigms.
We try to slightly modify the STE by adding a bias. The little amendment has little impact on RD-performance improvement, and We do not clearly distinguish our STE from minnen's in our paper.
If we have confused you, we sincerely apologize.


### Datasets
#### Training
Following the process in **STF**, you can download [OpenImages](https://github.com/openimages) for training, and the download script is available at [STF Repo](https://github.com/Googolxx/STF) in `downloader_openimages.py`.

#### Evaluation
We select [Kodak](https://r0k.us/graphics/kodak/), [CLIC.p](https://data.vision.ee.ethz.ch/cvl/clic/professional_valid_2020.zip), [CLIC.m](https://data.vision.ee.ethz.ch/cvl/clic/mobile_valid_2020.zip) as evaluation datasets(widely used ones). 

### Usage
The code is run with `python 3.9`, `pytorch 1.13.0`, `Compressai 1.2.6`, `timm 0.9.5`


#### Training
Personally speaking, I prefer not to add parameters in command lines, so the argv implemented one is not provided. 
Our `train.py` script largely follows the script in [DLPR](https://github.com/BYchao100/Deep-Lossy-Plus-Residual-Coding).
Notably, we save checkpoint files with `r'ckp{\lambda}.tar'` format for evaluation

#### Evaluation
We modify the eval script in [Compressai==1.2.6](https://github.com/InterDigitalInc/CompressAI/blob/master/compressai/utils/eval_model/__main__.py) for more cozy batch test.
The evaluation results are saved in xlsx file. And if you want to use our script, please follow the checkpoint file name format. 


## Related links
* CompressAI: https://github.com/InterDigitalInc/CompressAI