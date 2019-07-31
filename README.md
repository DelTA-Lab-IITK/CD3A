# Curriculum based Dropout Discriminator for Domain Adaptation(CD3A)

Torch code for Domain Adaptation model(CD3A) . For more information, please refer the [paper](https://arxiv.org/pdf/1907.10628.pdf) 

Accepted at [[BMVC 2019](https://bmvc2019.org)]

#####  [[Project]](https://delta-lab-iitk.github.io/CD3A//)     [[Paper Link ]](https://arxiv.org/pdf/1907.10628.pdf)

#### Abstract 
Domain adaptation is essential to enable wide usage of deep learning based networks trained using large labeled datasets. Adversarial learning based techniques have shown their utility towards solving this problem using a discriminator that ensures source and target distributions are close. However, here we suggest that rather than using a point
estimate, it would be useful if a distribution based discriminator could be used to bridge this gap. This could be achieved using multiple classifiers or using traditional ensemble methods. In contrast, we suggest that a Monte Carlo dropout based ensemble discriminator could suffice to obtain the distribution based discriminator. Specifically, we propose a curriculum based dropout discriminator that gradually increases the variance of the sample based distribution and the corresponding reverse gradients are used to align the source and target feature representations. The detailed results and thorough ablation analysis show that our model outperforms state-of-the-art results.


### Requirements
This code is written in Lua and requires [Torch](http://torch.ch/). 


You also need to install the following package in order to sucessfully run the code.
- [Torch](http://torch.ch/docs/getting-started.html#_)
- [loadcaffe](https://github.com/szagoruyko/loadcaffe)


#### Download Dataset
- [Office -31](https://pan.baidu.com/s/1o8igXT4)
- [ImageClef](https://pan.baidu.com/s/1lx2u1SMlSamsHnAPWrAHWA)
- [Office-Home](http://hemanthdv.org/OfficeHome-Dataset/)

##### Prepare Datasets
- Download the dataset


### Training Steps

We have prepared everything for you ;)

####Clone the repositotry 

``` git clone https://github.com/vinodkkurmi/DiscriminatorDomainAdaptation  ```

#### Dataset prepare
- Downalod dataset

-  put all source images inside mydataset/train/ such that folder name is class name
```
  mkdir -p /path_to_wherever_you_want/mydataset/train/ 
```
- put all target images inside mydataset/val/ such that folder name is class name

``` 
mkdir -p /path_to_wherever_you_want/mydataset/val/ 
```
- creare softlink of dataset
```
 cd DiscriminatorDomainAdaptation/
 ln -sf /path_to_wherever_you_want/mydataset dataset
```
 
  

#### Pretrained Alexnet model
- Download Alexnet pretraine caffe model [Link](https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet)

``` 
cd DiscriminatorDomainAdaptation/  
```

```
ln -sf /path_to_where_model_is_downloaded/ pretrained_network 
```

#### Train model
``` 
cd CD3A/  
./train.sh 
```




### Reference

If you use this code as part of any published research, please acknowledge the following paper

```
@article{kurmi2019curriculum,
  title={Curriculum based Dropout Discriminator for Domain Adaptation},
  author={Kurmi, Vinod Kumar and Bajaj, Vipul and Subramanian, Venkatesh K and Namboodiri, Vinay P},
  journal={arXiv preprint arXiv:1907.10628},
  year={2019}
}
```

## Contributors
* [Vinod K. Kurmi][1] (vinodkk@iitk.ac.in)



[1]: https://github.com/vinodkkurmi




