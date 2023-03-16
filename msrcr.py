import numpy as np
import cv2


def retinex_FM(img, iter=4):
    '''log(OP(x,y))=1/2{log(OP(x,y))+[log(OP(xs,ys))+log(R(x,y))-log(R(xs,ys))]*}, see
       matlab code in https://www.cs.sfu.ca/~colour/publications/IST-2000/'''
    if len(img.shape) == 2:
        img = img[..., None]
    ret = np.zeros(img.shape, dtype='uint8')

    def update_OP(x, y):
        nonlocal OP
        IP = OP.copy()
        if x > 0 and y == 0:
            IP[:-x, :] = OP[x:, :] + R[:-x, :] - R[x:, :]
        if x == 0 and y > 0:
            IP[:, y:] = OP[:, :-y] + R[:, y:] - R[:, :-y]
        if x < 0 and y == 0:
            IP[-x:, :] = OP[:x, :] + R[-x:, :] - R[:x, :]
        if x == 0 and y < 0:
            IP[:, :y] = OP[:, -y:] + R[:, :y] - R[:, -y:]
        IP[IP > maximum] = maximum
        OP = (OP + IP) / 2

    for i in range(img.shape[-1]):
        R = np.log(img[..., i].astype('double') + 1)
        maximum = np.max(R)
        OP = maximum * np.ones(R.shape)
        S = 2 ** (int(np.log2(np.min(R.shape)) - 1))
        while abs(S) >= 1:  # iterations are slow
            for k in range(iter):
                update_OP(S, 0)
                update_OP(0, S)
            S = int(-S / 2)
        OP = np.exp(OP)
        mmin = np.min(OP)
        mmax = np.max(OP)
        ret[..., i] = (OP - mmin) / (mmax - mmin) * 255
    return ret.squeeze()


def get_gauss_kernel(sigma, dim=2):
    '''1D gaussian function: G(x)=1/(sqrt{2π}σ)exp{-(x-μ)²/2σ²}. Herein, μ:=0, after
       normalizing the 1D kernel, we can get 2D kernel version by
       matmul(1D_kernel',1D_kernel), having same sigma in both directions. Note that
       if you want to blur one image with a 2-D gaussian filter, you should separate
       it into two steps(i.e. separate the 2-D filter into two 1-D filter, one column
       filter, one row filter): 1) blur image with first column filter, 2) blur the
       result image of 1) with the second row filter. Analyse the time complexity: if
       m&n is the shape of image, p&q is the size of 2-D filter, bluring image with
       2-D filter takes O(mnpq), but two-step method takes O(pmn+qmn)'''
    ksize = int(np.floor(sigma * 6) / 2) * 2 + 1  # kernel size("3-σ"法则) refer to
    # https://github.com/upcAutoLang/MSRCR-Restoration/blob/master/src/MSRCR.cpp
    k_1D = np.arange(ksize) - ksize // 2
    k_1D = np.exp(-k_1D ** 2 / (2 * sigma ** 2))
    k_1D = k_1D / np.sum(k_1D)
    if dim == 1:
        return k_1D
    elif dim == 2:
        return k_1D[:, None].dot(k_1D.reshape(1, -1))


def gauss_blur_original(img, sigma):
    '''suitable for 1 or 3 channel image'''
    row_filter = get_gauss_kernel(sigma, 1)
    t = cv2.filter2D(img, -1, row_filter[..., None])
    return cv2.filter2D(t, -1, row_filter.reshape(1, -1))


def gauss_blur_recursive(img, sigma):
    '''refer to “Recursive implementation of the Gaussian filter”
       (doi: 10.1016/0165-1684(95)00020-E). Paper considers it faster than
       FFT(Fast Fourier Transform) implementation of a Gaussian filter.
       Suitable for 1 or 3 channel image'''
    pass


def gauss_blur(img, sigma, method='original'):
    if method == 'original':
        return gauss_blur_original(img, sigma)
    elif method == 'recursive':
        return gauss_blur_recursive(img, sigma)


def MultiScaleRetinex(img, sigmas=[15, 80, 250], weights=None, flag=True):
    '''equal to func retinex_MSR, just remove the outer for-loop. Practice has proven
       that when MSR used in MSRCR or Gimp, we should add stretch step, otherwise the
       result color may be dim. But it's up to you, if you select to neglect stretch,
       set flag as False, have fun'''
    if weights == None:
        weights = np.ones(len(sigmas)) / len(sigmas)
    elif not abs(sum(weights) - 1) < 0.00001:
        raise ValueError('sum of weights must be 1!')
    r = np.zeros(img.shape, dtype='double')
    img = img.astype('double')
    for i, sigma in enumerate(sigmas):
        r += (np.log(img + 1) - np.log(gauss_blur(img, sigma) + 1)) * weights[i]
    if flag:
        mmin = np.min(r, axis=(0, 1), keepdims=True)
        mmax = np.max(r, axis=(0, 1), keepdims=True)
        r = (r - mmin) / (mmax - mmin) * 255  # maybe indispensable when used in MSRCR or Gimp, make pic vibrant
        r = r.astype('uint8')
    return r


def retinex_AMSR(img, sigmas=[12, 80, 250]):
    '''see Proposed Method ii in "An automated multi Scale Retinex with Color
       Restoration for image enhancement"(doi: 10.1109/NCC.2012.6176791)'''
    img = img.astype('double') + 1  #
    msr = MultiScaleRetinex(img - 1, sigmas, flag=False)  #
    y = 0.05
    for i in range(msr.shape[-1]):
        v, c = np.unique((msr[..., i] * 100).astype('int'), return_counts=True)
        sort_v_index = np.argsort(v)
        sort_v, sort_c = v[sort_v_index], c[sort_v_index]  # plot hist
        zero_ind = np.where(sort_v == 0)[0][0]
        zero_c = sort_c[zero_ind]
        #
        _ = np.where(sort_c[:zero_ind] <= zero_c * y)[0]
        if len(_) == 0:
            low_ind = 0
        else:
            low_ind = _[-1]
        _ = np.where(sort_c[zero_ind + 1:] <= zero_c * y)[0]
        if len(_) == 0:
            up_ind = len(sort_c) - 1
        else:
            up_ind = _[0] + zero_ind + 1
        #
        low_v, up_v = sort_v[[low_ind, up_ind]] / 100  # low clip value and up clip value
        msr[..., i] = np.maximum(np.minimum(msr[:, :, i], up_v), low_v)
        mmin = np.min(msr[..., i])
        mmax = np.max(msr[..., i])
        msr[..., i] = (msr[..., i] - mmin) / (mmax - mmin) * 255
    msr = msr.astype('uint8')
    return msr
    '''step of color restoration, maybe all right
    r=(np.log(125*img)-np.log(np.sum(img,axis=2))[...,None])*msr
    mmin,mmax=np.min(r),np.max(r)
    return ((r-mmin)/(mmax-mmin)*255).astype('uint8')
    '''
