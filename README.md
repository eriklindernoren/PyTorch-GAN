# PyTorch-GAN

## About
Collection of PyTorch implementations of Generative Adversarial Networks (GANs) suggested in research papers. These models are in some cases simplified versions of the ones ultimately described in the papers, but I have chosen to focus on getting the core ideas covered instead of getting every layer configuration right. Contributions and suggestions of GAN varieties to implement are very welcomed.

See also: [Keras-GAN](https://github.com/eriklindernoren/Keras-GAN)

## Table of Contents
- [Keras-GAN](#keras-gan)
  * [About](#about)
  * [Table of Contents](#table-of-contents)
  * [Installation](#installation)
  * [Implementations](#implementations)
    + [Auxiliary Classifier GAN](#ac-gan)
    + [Adversarial Autoencoder](#adversarial-autoencoder)
    + [Boundary-Seeking GAN](#bgan)
    + [Conditional GAN](#cgan)
    + [Context-Conditional GAN](#cc-gan)
    + [CycleGAN](#cyclegan)
    + [Deep Convolutional GAN](#dcgan)
    + [DiscoGAN](#discogan)
    + [Generative Adversarial Network](#gan)
    + [LSGAN](#lsgan)
    + [Pix2Pix](#pix2pix)
    + [Semi-Supervised GAN](#sgan)
    + [Super-Resolution GAN](#srgan)
    + [Wasserstein GAN](#wgan)

## Installation
    $ git clone https://github.com/eriklindernoren/PyTorch-GAN
    $ cd PyTorch-GAN/
    $ sudo pip3 install -r requirements.txt

## Implementations   
### AC-GAN
_Auxiliary Classifier Generative Adversarial Network_

[[Paper]](https://arxiv.org/abs/1610.09585) [[Code]](implementations/acgan/acgan.py)

#### Run Example
```
$ cd implementations/acgan/
$ python3 acgan.py
```

### Adversarial Autoencoder
_Adversarial Autoencoder_

[[Paper]](https://arxiv.org/abs/1511.05644) [[Code]](implementations/aae/adversarial_autoencoder.py)

#### Run Example
```
$ cd implementations/aae/
$ python3 aae.py
```

### BGAN
_Boundary-Seeking Generative Adversarial Networks_

[[Paper]](https://arxiv.org/abs/1702.08431) [[Code]](implementations/bgan/bgan.py)

#### Run Example
```
$ cd implementations/bgan/
$ python3 bgan.py
```


### CC-GAN
_Semi-Supervised Learning with Context-Conditional Generative Adversarial Networks_

[[Paper]](https://arxiv.org/abs/1611.06430) [[Code]](implementations/ccgan/ccgan.py)

#### Run Example
```
$ cd implementations/ccgan/
$ python3 ccgan.py
```

### CGAN
_Conditional Generative Adversarial Nets_

[[Paper]](https://arxiv.org/abs/1411.1784) [[Code]](implementations/cgan/cgan.py)

#### Run Example
```
$ cd implementations/cgan/
$ python3 cgan.py
```


### CycleGAN
_Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks_

[[Paper]](https://arxiv.org/abs/1703.10593) [[Code]](implementations/cyclegan/cyclegan.py)

<p align="center">
    <img src="http://eriklindernoren.se/images/cyclegan.png" width="640"\>
</p>

#### Run Example
```
$ cd data/
$ bash download_cyclegan_dataset.sh apple2orange
$ cd ../implementations/cyclegan/
$ python3 cyclegan.py
```   

### DCGAN
_Deep Convolutional Generative Adversarial Network_

[[Paper]](https://arxiv.org/abs/1511.06434) [[Code]](implementations/dcgan/dcgan.py)

#### Run Example
```
$ cd implementations/dcgan/
$ python3 dcgan.py
```

### DiscoGAN
_Learning to Discover Cross-Domain Relations with Generative Adversarial Networks_

[[Paper]](https://arxiv.org/abs/1703.05192) [[Code]](implementationsdiscogan/discogan.py)

<p align="center">
    <img src="http://eriklindernoren.se/images/discogan_architecture.png" width="640"\>
</p>

#### Run Example
```
$ cd data/
$ bash download_pix2pix_dataset.sh edges2shoes
$ cd ../implementations/discogan/
$ python3 discogan.py
```


### GAN
_Generative Adversarial Network_ with a MLP generator and discriminator

[[Paper]](https://arxiv.org/abs/1406.2661) [[Code]](implementations/gan/gan.py)

#### Run Example
```
$ cd implementations/gan/
$ python3 gan.py
```

### LSGAN
_Least Squares Generative Adversarial Networks_

[[Paper]](https://arxiv.org/abs/1611.04076) [[Code]](implementations/lsgan/lsgan.py)

#### Run Example
```
$ cd implementations/lsgan/
$ python3 lsgan.py
```

### Pix2Pix
_Unpaired Image-to-Image Translation with Conditional Adversarial Networks_

[[Paper]](https://arxiv.org/abs/1611.07004) [[Code]](implementations/pix2pix/pix2pix.py)

<p align="center">
    <img src="http://eriklindernoren.se/images/pix2pix_architecture.png" width="640"\>
</p>

#### Run Example
```
$ cd data/
$ bash download_pix2pix_dataset.sh facades
$ cd ../implementations/pix2pix/
$ python3 pix2pix.py
```  

### SGAN
_Semi-Supervised Generative Adversarial Network_

[[Paper]](https://arxiv.org/abs/1606.01583) [[Code]](implementations/sgan/sgan.py)

#### Run Example
```
$ cd implementations/sgan/
$ python3 sgan.py
```

### SRGAN
_Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network_

[[Paper]](https://arxiv.org/abs/1609.04802) [[Code]](implementations/srgan/srgan.py)

<p align="center">
    <img src="http://eriklindernoren.se/images/superresgan.png" width="640"\>
</p>

#### Run Example
```
$ cd implementations/srgan/
<follow steps at the top of srgan.py>
$ python3 srgan.py
```

### WGAN
_Wasserstein GAN_ (with DCGAN generator and discriminator)

[[Paper]](https://arxiv.org/abs/1701.07875) [[Code]](implementations/wgan/wgan.py)

#### Run Example
```
$ cd implementations/wgan/
$ python3 wgan.py
```
