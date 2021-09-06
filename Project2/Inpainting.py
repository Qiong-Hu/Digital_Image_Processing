from skimage import io, color, util, metrics
from scipy.signal import convolve2d as conv
from sklearn.preprocessing import normalize
from sklearn.linear_model import orthogonal_mp
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Normalize any matrix to the range of [0, 1]
def norm(x):
    x = np.array(x)
    x_min = np.min(x)
    x_max = np.max(x)
    x = (x - x_min)/(x_max - x_min)
    return x

# Prob 2.1
im_ori = io.imread('163096.jpg')
im_gray = color.rgb2gray(im_ori)
im_train = im_gray[30:286, 0:256]

im_ori = io.imread('113044.jpg')
im_gray = color.rgb2gray(im_ori)
im_test = im_gray[10:266, 150:406]

# Add masking texts
texts = 'ECE211A HW2'
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
color = 1
thickness = 1
im_test_missing = im_test.copy()
for i in range(8):
    cv2.putText(im_test_missing, texts, (20, 35*i+35), font, fontScale, color, thickness, cv2.LINE_AA)

im_mask = np.floor(norm(im_test - im_test_missing))


# # PSNR, SSIM
# print(metrics.peak_signal_noise_ratio(im_test, im_test_missing, data_range = 1))
# print(metrics.structural_similarity(im_test, im_test_missing, data_range = 1, gaussian_weights = True, sigma = 1.5, use_sample_covariance = False))


# Prob 3.1
# fig, ax = plt.subplots(5, 5, figsize = (6, 6))
# fig2, ax2 = plt.subplots(5, 5, figsize = (6, 6))
# for i in range(5):
#     for j in range(5):
#         rand = [np.random.randint(0, 32), np.random.randint(0, 32)]
#         # print(i, j, rand)
#         patch = im_test_missing[rand[0]*8:rand[0]*8+8, rand[1]*8:rand[1]*8+8]
#         mask_patch = im_mask[rand[0]*8:rand[0]*8+8, rand[1]*8:rand[1]*8+8]
#         ax[i][j].imshow(patch, plt.cm.gray)
#         ax[i][j].set_xticklabels([])
#         ax[i][j].set_yticklabels([])
#         ax2[i][j].imshow(mask_patch, plt.cm.gray)
#         ax2[i][j].set_xticklabels([])
#         ax2[i][j].set_yticklabels([])
# fig.suptitle('Testing Image Missing Patches', fontsize = 8, y = 1)
# fig2.suptitle('Testing Image Missing Patches Mask', fontsize = 8, y = 1)
# fig.tight_layout()
# fig2.tight_layout()
# plt.show()
# fig.savefig("im_test_missing_patches_title.jpg", dpi = 96, bbox_inches = 'tight')
# fig2.savefig("im_mask_missing_patches_title.jpg", dpi = 96, bbox_inches = 'tight')


# Prob 3.2
# fig, ax = plt.subplots(5, 5, figsize = (6, 6))
# for i in range(5):
#     for j in range(5):
#         dic_rand = np.random.random((8,8))
#         ax[i][j].imshow(dic_rand, plt.cm.gray)
#         ax[i][j].set_xticklabels([])
#         ax[i][j].set_yticklabels([])
# fig.suptitle('Random Dictionary Atoms', fontsize = 8, y = 1)
# fig.tight_layout()
# plt.show()
# fig.savefig("im_random_dict_title.jpg", dpi = 96, bbox_inches = 'tight')


# Prob 3.3
def reshape(img):
    # img: (256, 256), original image
    # img_reshape: (64, 1024), without normalization
    # a column in img_reshape = a patch in img
    img2 = img.copy()
    img_reshape = img2.reshape(64, 1024)
    y = img_reshape.copy()
    for i in range(32):
        for j in range(32):
            patch = img2[i*8:i*8+8, j*8:j*8+8]
            pitch = patch.flat
            img_reshape[:, i*32+j] = pitch
    return img_reshape

def reconstruct(dictionary, y, im_mask, n_nonzero_coefs = 20):
    im_recons = np.zeros(y.shape)
    for i in range(32):
        for j in range(32):
            patch = y[i*8:i*8+8, j*8:j*8+8].flat
            mask_patch = im_mask[i*8:i*8+8, j*8:j*8+8].flat

            index = np.nonzero(mask_patch)[0]
            patch_mean = np.mean(patch[index])
            patch_norm = np.linalg.norm(patch[index] - patch_mean)
            patch_normalized = (patch[index] - patch_mean)/patch_norm

            sparse_code = orthogonal_mp(dictionary[index, :], patch_normalized, n_nonzero_coefs = n_nonzero_coefs)
            recons_patch = dictionary.dot(sparse_code)
            recons_patch = recons_patch * patch_norm + patch_mean
            recons_patch = recons_patch.reshape(8, 8)
            im_recons[i*8:i*8+8, j*8:j*8+8] = recons_patch[:, :]
    im_recons = norm(im_recons)
    return im_recons


# Prob 3.4
# n_atom = 512
# dic_rand = np.random.random((64, n_atom))
# dic_rand = normalize(dic_rand, axis = 0)
# im_recons_rand = reconstruct(dic_rand, im_test_missing, im_mask)


# # PSNR, SSIM
# print(metrics.peak_signal_noise_ratio(im_test, im_recons_rand, data_range = 1))
# print(metrics.structural_similarity(im_test, im_recons_rand, data_range = 1, gaussian_weights = True, sigma = 1.5, use_sample_covariance = False))


# Prob 4.1
# K-SVD
def dict_update(y, d, x, n_atom = 512):
    # y: img, normalized
    # d: dictionary, unit normalized for each column
    # x: sparse code
    # n_atom: atom number of the dictionary
    for k in range(n_atom):
        index = np.nonzero(x[k,:])[0]   # index of support of residue
        if len(index) == 0:
            continue
        # Update the k-th column of the dictionary
        d[:, k] = 0
        # Residue
        r = (y - np.dot(d, x))[:, index]
        u, s, v = np.linalg.svd(r, full_matrices = False)
        d[:, k] = u[:, 0]
        x[k, index] = s[0] * v[0, :]
    return d, x


# Prob 4.2
n_atom = 512
dictionary = np.random.random((64, n_atom))
dictionary = normalize(dictionary, axis = 0)
y_train = reshape(im_train)
for i in range(0):     # Max loop = 20
    x = orthogonal_mp(dictionary, y_train, n_nonzero_coefs = 20)
    e = np.linalg.norm(y_train - np.dot(dictionary, x))
    dictionary, x = dict_update(y_train, dictionary, x, n_atom)
    print("Loop " + str(i) + ": error = " + str(e))

# Picture dictionary (25 smallest std columns)
dict_std = np.std(dictionary, axis = 0)
dict_std_index = np.argsort(dict_std)

# fig, ax = plt.subplots(5, 5, figsize = (6, 6))
# for i in range(5):
#     for j in range(5):
#         index = i*5+j
#         dic_patch = dictionary[:, dict_std_index[index]].reshape((8, 8))
#         ax[i][j].imshow(dic_patch, plt.cm.gray)
#         ax[i][j].set_xticklabels([])
#         ax[i][j].set_yticklabels([])
# fig.suptitle('K-SVD Dictionary Atoms', fontsize = 8, y = 1)
# fig.tight_layout()
# # plt.show()
# fig.savefig("im_ksvd_dict_title.jpg", dpi = 96, bbox_inches = 'tight')


# Prob 4.3
im_recons = reconstruct(dictionary, im_test_missing, im_mask)


# # PSNR, SSIM
# print(metrics.peak_signal_noise_ratio(im_test, im_recons, data_range = 1))
# print(metrics.structural_similarity(im_test, im_recons, data_range = 1, gaussian_weights = True, sigma = 1.5, use_sample_covariance = False))



# # For quick test
# io.imshow(im_recons, cmap = 'gray')
# io.show()
# io.imsave('im.jpg', im_recons)

np.set_printoptions(threshold = np.infty)
np.set_printoptions(precision = 3)
np.set_printoptions(suppress = True)


# For img output
fig, ax = plt.subplots(1, 1)
ax.imshow(im_recons, plt.cm.gray)
ax.set_title('Reconstructed Image with K-SVD Dictionary', fontsize = 8)
ax.set_xticklabels(np.uint8(np.linspace(0,300,7))-50, fontsize = 8)
ax.set_yticklabels(np.uint8(np.linspace(0,300,7))-50, fontsize = 8)
plt.tight_layout()
# plt.show()
# fig.savefig("im_title.jpg", dpi = 96, bbox_inches = 'tight')
