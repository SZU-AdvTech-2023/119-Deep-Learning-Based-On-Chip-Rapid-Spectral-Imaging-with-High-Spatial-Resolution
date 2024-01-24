"""
Performing correct sampling on hyperspectral images to maintain the intergral intergrity of the data
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import cv2
from pathlib import Path
from hsi_io import *
from path import valid_mask
from typing import Iterable

default_backend = 'numpy'
device = None
try:
    import torch
    if torch.cuda.is_available():
        default_backend = 'torch'
        device = torch.device('cuda')
except:
    pass


def patch_sample(img, patch_size, num_samples, mask=None)->Iterable[np.ndarray]:
    """
    Sample patches from image within the mask if given
    :param img: input image
    :param patch_size: patch size
    :param num_samples: number of samples
    :param mask: mask of the image, default is None
    :return: patches
    """
    img_size = img.shape
    if mask is None:
        mask = np.ones(img_size[0:2], dtype=np.bool)
    else:
        mask = mask.astype(bool)

    # find the bounding box of the mask
    points = np.argwhere(mask)
    x_min = np.min(points[:, 0])
    x_max = np.max(points[:, 0])
    y_min = np.min(points[:, 1])
    y_max = np.max(points[:, 1])

    # sample patches
    patches = []
    i=0
    while i < num_samples:
        x = np.random.randint(x_min, x_max - patch_size[0] + 1)
        y = np.random.randint(y_min, y_max - patch_size[1] + 1)
        if mask[x, y]:
            patch = img[x:x + patch_size[0], y:y + patch_size[1], :]

            if patch.min() < 1e-6:
                continue
            patches.append(patch.reshape(patch_size[0]* patch_size[1], -1).mean(axis=0))
            i+=1

    patches = np.array(patches)
    return patches

def upsampling_matrix(original_wl,target_wl=np.arange(400,1000.5,0.5)):
    """
    Upsample the data from original_wl to target_wl using linear interpolation
    :param data: input data
    :param original_wl: original wavelength
    :param target_wl: target wavelength
    :return: upsampled data
    """
    interp_matrix = np.zeros((original_wl.shape[0], target_wl.shape[0]))
    for i in range(original_wl.shape[0]):
        interp_matrix[i] = np.interp(target_wl, original_wl, np.eye(original_wl.shape[0])[i])
    return interp_matrix

def downsampling_matrix(original_wl,target_wl):
    """
    Generate downsampling matrix from original_wl to target_wl using convolution
    :param original_wl: original wavelength
    :param target_wl: target wavelength
    """
    conv_matrix = np.zeros((original_wl.shape[0],original_wl.shape[0]))
    original_step = original_wl[1]-original_wl[0]
    target_step = target_wl[1]-target_wl[0]
    kernel_size = int(np.ceil(target_step/original_step))+1
    kernel = np.ones(kernel_size)
    kernel = kernel/np.sum(kernel)

    for i in range(original_wl.shape[0]):
        conv_matrix[:,i] = np.convolve(np.eye(original_wl.shape[0])[i],kernel,mode='same')


    # boundary compensation
    for i in range(int(np.floor(kernel_size/2))):
        conv_matrix[0,i] += max(1- conv_matrix[:,i].sum() ,0)
        conv_matrix[-1,-i-1] += max(1- conv_matrix[:,-i-1].sum() ,0)

    # remove the redundant bands
    min_index = np.argmin(np.abs(original_wl-target_wl[0]))
    max_index = np.argmin(np.abs(original_wl-target_wl[-1]))
    
    
    return conv_matrix[:, min_index:max_index+1:kernel_size-1]

def apply_sampling_matrix(img:np.ndarray, sampling_matrix:np.ndarray, backend='numpy')->np.ndarray:
    """
    Apply sampling matrix to image in spectral dimension
    :param img: input image, [h,w,n_in] or [n_px,n_in]
    :param sampling_matrix: sampling matrix [n_in, n_out]
    :return: sampled image
    """
    assert img.shape[-1] == sampling_matrix.shape[0], "image and sampling matrix should have same number of bands"
    if backend == 'numpy':
        if len(img.shape) == 2:
            return np.dot(img, sampling_matrix)
        return np.einsum('ijk,kl->ijl', img, sampling_matrix)
    elif backend == 'torch':
        img = torch.from_numpy(img).float().to(device)
        sampling_matrix = torch.from_numpy(sampling_matrix).float().to(device)
        if len(img.shape) == 2:
            return torch.matmul(img, sampling_matrix).cpu().numpy()
        return torch.matmul(img, sampling_matrix).cpu().numpy()

mask = plt.imread(valid_mask)[:,:,0].astype(bool)


if __name__=='__main__':
    # load envi image begin with R_ in a folder
    target_folder = Path(r'\\Product-NAS\高光谱测试样本库\高光谱数据集2023\室外')
    target_files = list(target_folder.glob('**/R_*.hdr'))
    target_files.sort()

    to_folder = Path(r'\\Product-NAS\高光谱测试样本库\高光谱数据集2023\temp\extracted')

    wl = None
    for idx, hdrpath in enumerate(target_files):
        try:
            print(idx, hdrpath)
            hdrf = load_envi_img(hdrpath)
            wl = np.array(hdrf.metadata['wavelength'],dtype=float)
            img = hdrf.load()

            wl_indexs = np.arange(0,wl.shape[0])
            # remove the O-3 absorption band
            wl_indexs = np.delete(wl_indexs,np.arange(505,515))
            wl = np.delete(wl,np.arange(505,515))
            img = img[:,:,wl_indexs]
            
            patches = patch_sample(img, (3, 3), 10000,mask=mask)

            # since the original data is not equally spaced, we need to interpolate it to equally spaced by upsampling first
            up_mat = upsampling_matrix(wl,np.arange(400,1000.5,0.5))
            # then we can downsample it to the target wavelength
            down_mat = downsampling_matrix(np.arange(400,1000.5,0.5),np.arange(400,1005,5))
            # combine the two matrix
            interp_mat = up_mat@down_mat

            patches = apply_sampling_matrix(patches,interp_mat)

            save_path = to_folder/hdrpath.with_suffix('.mat').name
            sio.savemat(save_path,{'data':patches,'wavelength':np.arange(400,1005,5)})
        except Exception as e:
            print(e)
            continue
        