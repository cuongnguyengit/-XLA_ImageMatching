from skimage import feature
import numpy as np, matplotlib.pyplot as plt
from skimage.feature import greycomatrix, greycoprops, local_binary_pattern
import scipy.stats
import cv2


class Feature:
    @staticmethod
    def featureHog(gray):
        cell_size = (8, 8)  # h x w in pixels
        block_size = (2, 2)  # h x w in cells
        nbins = 9  # number of orientation bins
        H = feature.hog(gray, orientations=nbins, pixels_per_cell=cell_size,
                        cells_per_block=block_size, transform_sqrt=True, block_norm="L2")
        return H

    @staticmethod
    def fea_color_moment(image):
        b, g, r = cv2.split(image)
        def f(x):
            # mean
            mean = np.mean(x)
            # variance
            var = np.var(x)
            # skew
            skew = scipy.stats.skew(np.concatenate(x))

            return [mean, var, skew]
        result = f(b)
        result.extend(f(g))
        result.extend(f(r))
        return np.array(result)


    @staticmethod
    def featureColorHist(gray):
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        return hist

    @staticmethod
    def fea_hu_moments(gray):
        # chuyển về ảnh gray
        M = cv2.moments(gray)
        # the spatial moments
        # the central moments
        # the normalized central moments
        feature = cv2.HuMoments(M).flatten()
        # 7 hu Moments - describe shape image
        return M, feature

    @staticmethod
    def fea_texture_comatrix(gray):
        Grauwertmatrix = greycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=False, normed=False)
        lf = []
        props = ['dissimilarity', 'contrast', 'homogeneity', 'energy', 'correlation']
        for f in props:
            lf.append(greycoprops(Grauwertmatrix, f)[0, 0])
        return np.array(lf)

    @staticmethod
    def fea_LBP(gray, numPoints=24, radius=8, eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = local_binary_pattern(gray, numPoints,
                                   radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, numPoints + 3),
                                 range=(0, numPoints + 2))
        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        # return the histogram of Local Binary Patterns
        return hist


def feature_all(path, hog=True, colorHist=True, moments=True, texture_comatrix=True, lbp=True, colorMoment=True):
    img = plt.imread(path, cv2.IMREAD_UNCHANGED)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(src=gray, dsize=(64, 128))
    array_fea = np.array([])
    if hog:
        temp = Feature.featureHog(gray).flatten()
        array_fea = np.hstack((array_fea, temp))
    if colorHist:
        temp = Feature.featureColorHist(gray).flatten()
        array_fea = np.hstack((array_fea, temp))
    if colorMoment:
        temp = Feature.fea_color_moment(img).flatten()
        array_fea = np.hstack((array_fea, temp))
    if moments:
        m, hu = Feature.fea_hu_moments(gray)
        m = np.array([v for k, v in m.items()])
        hu = hu.flatten()
        array_fea = np.hstack((array_fea, m))
        array_fea = np.hstack((array_fea, hu))
    if texture_comatrix:
        temp = Feature.fea_texture_comatrix(gray).flatten()
        array_fea = np.hstack((array_fea, temp))
    if lbp:
        temp = Feature.fea_LBP(gray, numPoints=24, radius=8).flatten()
        array_fea = np.hstack((array_fea, temp))
    return array_fea.reshape(1, array_fea.shape[0])


if __name__ == '__main__':
    img = plt.imread('Data/person.jpg', cv2.IMREAD_UNCHANGED)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(src=gray, dsize=(64, 128))
    # HOG
    HOG_fea = Feature.featureHog(gray)
    # Color Histogram
    CH_fea = Feature.featureColorHist(gray)
    # 7 Hu moments and 24 Moments
    Mo, Hu = Feature.fea_hu_moments(gray)
    # texture comatrix
    list_fea_comatrix = Feature.fea_texture_comatrix(gray)
    # LBP fea with numpoints and radius
    lbp = Feature.fea_LBP(gray)
    # color moments
    CM_fea = Feature.fea_color_moment(img)
    print('HOG', len(HOG_fea))
    print('CH', len(CH_fea))
    print('CM', len(CM_fea))
    print('Mo', len(Mo))
    print('Hu Mo', len(Hu))
    print('Texture Comatrix', len(list_fea_comatrix))
    print('LBP', len(lbp))
    print('LEN', len(HOG_fea) + len(CH_fea) + len(CM_fea) + len(Mo) + len(Hu) + len(list_fea_comatrix) + len(lbp))

    fea_sum = feature_all(path='Data/person.jpg')
    print(fea_sum.shape)
