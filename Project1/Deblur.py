from skimage import io, color, util, metrics
from scipy.signal import convolve2d as conv
import numpy as np
import matplotlib.pyplot as plt

# Normalize any matrix to the range of [0, 1]
def normalize(x):
    x = np.array(x)
    x_min = np.min(x)
    x_max = np.max(x)
    x = (x - x_min)/(x_max - x_min)
    return x

# Prob 2.1
im_ori = io.imread('im_ori.jpg')
# im_ori = io.imread('8068.jpg')
# im_ori = io.imread('196062.jpg')
im_gray = color.rgb2gray(im_ori)
im_crop = im_gray[0:256, 100:356]

kernel_size = 21
kernel = np.identity(kernel_size)/kernel_size

noiseless = conv(im_crop, kernel)

noisy = conv(im_crop, kernel)
noisy += 0.01 * noisy.std() * np.random.standard_normal(noisy.shape)

image_size = im_crop.shape[0]
noisy_size = noisy.shape[0]

# Prob 2.2
groundtruth = im_gray[0:256, 100:356]       
# groundtruth = im_gray[0:256, 130:386]       # For 8068.jpg
# groundtruth = im_gray[0:256, 120:376]       # For 196062.jpg
shift = int((kernel_size - 1) / 2)
noiseless2 = noiseless[shift : shift + image_size, shift : shift + image_size]
noisy2 = noisy[shift : shift + image_size, shift : shift + image_size]

# # PSNR(x, noiseless), SSIM(x, noiseless), PSNR(x, noisy), SSIM(x, noisy)
# print(metrics.peak_signal_noise_ratio(groundtruth, noiseless2, data_range = 1))
# print(metrics.structural_similarity(groundtruth, noiseless2, data_range = 1, gaussian_weights = True, sigma = 1.5, use_sample_covariance = False))
# print(metrics.peak_signal_noise_ratio(groundtruth, noisy2, data_range = 1))
# print(metrics.structural_similarity(groundtruth, noisy2, data_range = 1, gaussian_weights = True, sigma = 1.5, use_sample_covariance = False))

# Prob 3.1
def naive_deconv(y, h):
    Y = np.fft.fft2(y)
    H = np.fft.fft2(h, s = y.shape)

    epsilon = 1e-10
    X = Y / (H + epsilon)
    x = np.fft.ifft2(X)
    x = np.real(x[0:image_size, 0:image_size])
    x = normalize(x)
    return x

# Prob 3.2
naive_noiseless = naive_deconv(noiseless, kernel)
naive_noisy = naive_deconv(noisy, kernel)

# # PSNR(x, noiseless), SSIM(x, noiseless), PSNR(x, noisy), SSIM(x, noisy)
# print(metrics.peak_signal_noise_ratio(groundtruth, naive_noiseless, data_range = 1))
# print(metrics.structural_similarity(groundtruth, naive_noiseless, data_range = 1, gaussian_weights = True, sigma = 1.5, use_sample_covariance = False))
# print(metrics.peak_signal_noise_ratio(groundtruth, naive_noisy, data_range = 1))
# print(metrics.structural_similarity(groundtruth, naive_noisy, data_range = 1, gaussian_weights = True, sigma = 1.5, use_sample_covariance = False))

# Prob 4.1
# Element-wise square value of a matrix
def square(x):
    result = np.multiply(np.abs(x), np.abs(x))
    return result

# Signal to noise ratio
def SNR(signal, noise):
    # signal, noise: in space domain
    X = np.fft.fft2(signal, s = noise.shape)
    X2 = square(X)

    N = np.fft.fft2(noise)
    N2 = square(N)
    snr = X2/N2
    return snr

def wiener_filter(y, h, snr):
    Y = np.fft.fft2(y)
    H = np.fft.fft2(h, s = y.shape)
    H2 = square(H)

    # Wiener filter
    epsilon = 1e-10
    G = H2/(H2 + 1/snr)
    X = Y * G/(H + epsilon)
    x = np.fft.ifft2(X)
    x = np.real(x[0:image_size, 0:image_size])
    x = normalize(x)
    return x

# Prob 4.2
wiener_exact = wiener_filter(noisy, kernel, SNR(groundtruth, noisy - noiseless))

# # PSNR, SSIM
# print(metrics.peak_signal_noise_ratio(groundtruth, wiener_exact, data_range = 1))
# print(metrics.structural_similarity(groundtruth, wiener_exact, data_range = 1, gaussian_weights = True, sigma = 1.5, use_sample_covariance = False))

# Prob 4.3
# Power spectral density
def PSD(x):
    X = np.fft.fft2(x, s = noisy.shape)
    X_shift = np.fft.fftshift(X)
    X2 = square(X_shift)
    return X2

image_power = np.log(PSD(groundtruth))
# image_power = normalize(image_power)
noise_power = np.log(PSD(noisy - noiseless))
# noise_power = normalize(noise_power)


# Prob 4.4
# print(image_power[135:143,135:143])
# fig, ax = plt.subplots(1, 1)
# ax.plot(np.linspace(0,275,276)-138, np.log(PSD(groundtruth))[138,:],'k')
# ax.plot(np.linspace(0,275,276)-138, np.log(PSD(noisy))[138,:],'b')

# # ax.plot(np.linspace(0,275,276)-138, image_power[:,138],'b')
# ax.plot(np.linspace(0,275,276)-138, np.abs(np.linspace(0,275,276)-138)**(-0.2)*20, 'r', linewidth = 2)
# ax.plot(np.linspace(0,275,276)-138, np.abs(np.linspace(0,275,276)-138)**(-0.6)*20, 'g', linewidth = 2)
# ax.set_title('Log-scale Image Spectral Density', fontsize = 8)
# ax.set_xlabel('ω1, ω2 index')
# ax.set_ylabel('Log-scale power density')
# ax.set_xticklabels([-200,-150,-100,-50,0,50,100,150], fontsize = 8)
# ax.set_yticklabels([-10,-5,0,5,15,20,25], fontsize = 8)
# ax.legend(['Groundtruth Image', 'Noisy Image'])
# plt.tight_layout()
# plt.show()
# fig.savefig("img_psd.jpg", dpi = 300, bbox_inches = 'tight')

snr = np.zeros(shape = noisy.shape)
shift = noisy_size / 2    # shift = 138
for i in range(noisy_size):
    for j in range(noisy_size):
        # snr[i,j] = 1e7/((i-shift)**2+(j-shift)**2+1e-10)
        snr[i,j] = 1e-1*((i-shift)**2+(j-shift)**2+1e-10)
        # snr[i,j] = 1e5*max(1/(abs(i-shift)+1e-10), 1/(abs(j-shift)+1e-10))
        # snr[i,j] = 1e5*((abs(i-shift)*shift+1e-10)**(-0.6)+(abs(j-shift)*shift+1e-10)**(-0.6))
        # snr[i,j] = 1e8*(abs(i-shift)*shift+1e-10)**(-0.6)*(abs(j-shift)*shift+1e-10)**(-0.6)
# print(snr)

result = wiener_filter(noisy, kernel, snr)
# result = np.log(snr)
# print(result)

# # PSNR, SSIM
print(metrics.peak_signal_noise_ratio(groundtruth, result, data_range = 1))
print(metrics.structural_similarity(groundtruth, result, data_range = 1, gaussian_weights = True, sigma = 1.5, use_sample_covariance = False))

# For quick test
# io.imshow(result, cmap = 'gray')
# io.show()
# io.imsave('wiener_approx.jpg', result)

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)



# For img output
fig, ax = plt.subplots(1, 1)
ax.imshow(result, plt.cm.gray)
ax.set_title('Wiener Deconv with approximated SNR', fontsize = 8)
ax.set_xticklabels(np.uint8(np.linspace(0,300,7))-50, fontsize = 8)
ax.set_yticklabels(np.uint8(np.linspace(0,300,7))-50, fontsize = 8)
plt.tight_layout()
# plt.show()
# fig.savefig("wiener_approx_title.jpg", dpi = 96, bbox_inches = 'tight')
