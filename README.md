# Blind Super-Resolution dataset generator
This notebook allows creating datasets of images downscaled by random kernels.
This is used for Blind Super-Resolution challenges where the downscaling kernel should be predicted.

----------
![](/figs/fig.JPG)
----------
This code was used to create the dataset for the paper:
KernelGAN: Blind Super-Resolution Kernel Estimation using an Internal-GAN (Bell-Kligler, Shocher, Irani)  
Repository of KernelGAN: https://github.com/sefibk/KernelGAN  
Project page: http://www.wisdom.weizmann.ac.il/~vision/kernelgan/

----------------------
The kernels are basically unisotropic gaussians but there is also an option for multiplicative noise that makes the deviate from a pure gaussian.
You can control the downscale factors, even changing aspect-ratio, and the probablistic properties of sampling the kernels from the first cell in the notebook.

----------------------
Eventually images and kernels are saved to a wanted path and you get a preview at the bottom of the notebook

