Keras VAEs and GANs
===

> Keras implementation of various deep generative networks such as VAE and GAN.

## Models

#### Standard models

* Variational autoencoder (VAE)
* Generative adversarial network (GAN or DCGAN)
* Energy-based GAN (EBGAN)
* Adversarially learned inference (ALI)

#### Conditional models

* Conditional variational autoencoder
* CVAE-GAN


## Usage

Example programs are included in ``examples`` directory. These programs can be used as

```shell
python train.py --model=dcgan --epoch=200 --batchsize=100 --output=output
```
