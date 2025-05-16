# GIP-Stereo : Geometry-Aware Information Propagation Network for Stereo Matching


Yang Zhao<sup>1</sup>, Ziyang Chen<sup>1</sup>, Junling He<sup>1</sup>, Wenting Li, Yao Xiao, Chunwei Tian, Yongjun Zhang*


~~~
The official implementation of "GIP-Stereo：Geometry-Aware Information Propagation Network for Stereo Matching".
This paper is under review processing... We will release all codes and results after finishing the review.
~~~

## Environment
* PyTorch 2.1.0
* CUDA 11.8
 ~~~
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
~~~

## Required Data
* [SceneFlow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
* [KITTI](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)
* [ETH3D](https://www.eth3d.net/datasets)
* [Middlebury](https://vision.middlebury.edu/stereo/submit3/)
* [TartanAir](https://github.com/castacks/tartanair_tools)
* [CREStereo Dataset](https://github.com/megvii-research/CREStereo)
* [FallingThings](https://research.nvidia.com/publication/2018-06_falling-things-synthetic-dataset-3d-object-detection-and-pose-estimation)
* [InStereo2K](https://github.com/YuhuaXu/StereoDataset)
* [Sintel Stereo](http://sintel.is.tue.mpg.de/stereo)
* [HR-VS](https://drive.google.com/file/d/1SgEIrH_IQTKJOToUwR1rx4-237sThUqX/view)

## Training 
### Training on Sceneflow:
~~~
./train.sh
~~~



## Acknowledgements
<ul>
<li>This project borrows the code from <strong><a href="https://github.com/gangweiX/IGEV">IGEV</a></strong>, <a href="https://github.com/David-Zhao-1997/High-frequency-Stereo-Matching-Network">DLNR</a>, <a href="https://github.com/princeton-vl/RAFT-Stereo">RAFT-Stereo</a>, <strong><a href="https://github.com/ZYangChen/MoCha-Stereo">MoCha</a>. We thank the original authors for their excellent works!</li>
</ul>
