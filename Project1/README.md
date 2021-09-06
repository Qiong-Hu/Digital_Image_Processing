# Project 1: Image Deblur using Deconvolution Algorithm

Main code: [Deconv.py](Deconv.py)

Full report: [report.pdf](HW1_report.pdf)

Python packages used in implementation: skimage, scipy, numpy, matplotlib

Image source: BSDS500 dataset

<br>

## Objective

To implement several deconvolution algorithms to recover the original image from the blurred observation, and compare the performance of them.

## Instructions

1. Generate blurred image from original image using 2D convolution and Gaussian noise.

	<img src="img/original_image.jpg" width="30%"> <img src="img/blurry_image.jpg" width="30%"> <img src="img/blurry_noisy_image.jpg" width="30%">

	(Left to right: original image, noiseless image, noisy image)

2. Apply naive deconvolution, Wiener deconvolution on blurred in frequency domain.

3. Estimate frequency-dependent Signal-Noise Ratio (SNR) by analyzing the power spectral density of images.

4. Experiment with different approximation of SNR function for Wiener Filter algorithm.

## Evaluation

Evaluation methods: two image similarity metrics: peak signal-to-noise ratio (PSNR) and structural similarity (SSIM) index.

The results are as follows:

| Deconvolution Algorithm | PSNR | SSIM |
|:-----------------------:|:----:|:----:|
|Baseline|16.6020|0.4104|
|Naïve Deconvolution|11.2365|0.0233|
|Wiener: known SNR|24.7730|0.8450|
|Wiener: <img src="https://render.githubusercontent.com/render/math?math=1/(\omega_1^2%2b\omega_2^2)">|22.7635|0.7459|
|Wiener: <img src="https://render.githubusercontent.com/render/math?math=\omega_1^2%2b\omega_2^2">|26.5565|**0.8084**|
|Wiener: <img src="https://render.githubusercontent.com/render/math?math=\max\left\{1/\omega_1,1/\omega_2\right\}">|24.4619|0.7193|
|Wiener: <img src="https://render.githubusercontent.com/render/math?math=\omega_1^{-0.6}%2b\omega_2^{-0.6}">|22.7547|0.7484|
|Wiener: <img src="https://render.githubusercontent.com/render/math?math=\omega_1^{-0.6}*\omega_2^{-0.6}">|23.9514|0.7409|

## Visualization

| SNR function | log-scale SNR image | deblurred image |
|:------------:|:-------------------:|:---------------:|
| Known SNR from original image | <img src="img/image_power.jpg"> | <img src="img/wiener_exact.jpg"> |
| <img src="https://render.githubusercontent.com/render/math?math=\LARGE\frac{1}{\omega%20_1^2%2b\omega%20_2^2}"> | <img src="img/SNR1.jpg"> | <img src="img/wiener_approx1.jpg"> |
| <img src="https://render.githubusercontent.com/render/math?math=\LARGE\omega%20_1^2%2b\omega%20_2^2"> | <img src="img/SNR2.jpg"> | <img src="img/wiener_approx2.jpg"> |
| <img src="https://render.githubusercontent.com/render/math?math=\LARGE\max\left\{\frac{1}{\omega%20_1},%20\frac{1}{\omega%20_2}\right\}"> | <img src="img/SNR3.jpg"> | <img src="img/wiener_approx3.jpg"> |
| <img src="https://render.githubusercontent.com/render/math?math=\LARGE\omega%20_1^{-0.6}%2b\omega%20_2^{-0.6}"> | <img src="img/SNR4.jpg"> | <img src="img/wiener_approx4.jpg"> |
| <img src="https://render.githubusercontent.com/render/math?math=\LARGE\omega%20_1^{-0.6}*\omega%20_2^{-0.6}"> | <img src="img/SNR5.jpg"> | <img src="img/wiener_approx5.jpg"> |

<br>

Full report see: [report.pdf](HW1_report.pdf)
