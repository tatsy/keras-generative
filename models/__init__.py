# Ordinary generative models
from .vae import VAE
from .dcgan import DCGAN
from .improvedgan import ImprovedGAN
from .ebgan import EBGAN
from .began import BEGAN
from .ali import ALI

# Conditional generative models
from .cvae import CVAE
from .cvaegan import CVAEGAN
from .cali import CALI
from .triplegan import TripleGAN

# Image-to-image genearative models
from .cyclegan import CycleGAN
from .unit import UNIT
