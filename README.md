Keras VAEs and GANs
===

Keras implementation of various deep generative networks such as VAE and GAN.

## Models

#### Standard models

* Variational autoencoder (VAE) [Kingma et al. 2013]
* Generative adversarial network (GAN or DCGAN) [Goodfellow et al. 2014]
* Energy-based GAN (EBGAN) [Zhao et al. 2016]
* Adversarially learned inference (ALI) [Dumoulin et al. 2017]

#### Conditional models

* Conditional variational autoencoder [Kingma et al. 2014]
* CVAE-GAN [Bao et al. 2017]

## Usage

Example programs are included in ``examples`` directory. These programs can be used as

```shell
# Standard models
python train.py --model=dcgan --epoch=200 --batchsize=100 --output=output

# Conditional models
python train_conditional.py --model=cvaegan --epoch=200 --batchsize=100 --output=output
```

## References

* Kingma et al., "Auto-Encoding Variational Bayes", arXiv preprint 2013.
* Goodfellow et al., "Generative adversarial nets", NIPS 2014.
* Zhao et al., "Energy-based generative adversarial network", arXiv preprint 2016.
* Dumoulin et al. "Adversarially learned inference", ICLR 2017.
* Kingma et al., "Semi-supervised learning with deep generative models", NIPS 2014.
* Bao et al., "CVAE-GAN: Fine-Grained Image Generation through Asymmetric Training", arXiv preprint 2017.
