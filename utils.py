import numpy as np
import scipy.stats.mstats
import matplotlib.pyplot as plt
import cv2
import os

def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def corr2_coeff(A, B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean()
    B_mB = B - B.mean()

    # Sum of squares across rows
    ssA = (A_mA ** 2).mean()
    ssB = (B_mB ** 2).mean()

    # Finally get corr coeff
    coef = (A_mA * B_mB).mean() / np.sqrt(ssA * ssB)
    return coef

def r2coef(gt, pred):
    gt_mean = gt.mean()
    r2coef = 1 - np.sum((gt - pred) ** 2) / np.sum((gt - gt_mean) ** 2)
    return r2coef

def get_rmse(gt, pred):
    return np.sqrt(np.mean((gt - pred) ** 2))

def get_kl(gt, pred, chance=0): # Kullback-Leibler divergence
    kl = np.sum(gt * np.where(gt > chance, np.log(gt), 0) - gt * np.where(pred >= chance, np.log(pred), 0))
    return kl

def get_spearmanr(gt, pred):
    try:
        return scipy.stats.spearmanr(gt, pred)[0]
    except:
        return 0


def label_accuracy(label_trues, label_preds):
    """Returns accuracy score evaluation result.
    """

    gt = label_trues.astype(np.float64)
    pred = label_preds.astype(np.float64)
    gt_1d, pred_1d = gt.ravel() / gt.sum(), pred.ravel() / pred.sum()
    gt_1d_01, pred_1d_01 = gt.ravel() / 255.0, pred.ravel() / 255.0

    cc = corr2_coeff(gt_1d, pred_1d)
    chance = 0.5
    kl = get_kl(gt_1d, pred_1d, chance=chance)
    kl_01 = get_kl(gt_1d_01, pred_1d_01, chance=chance)
#     spearman = get_spearmanr(gt_1d, pred_1d)
    r2 = r2coef(gt_1d, pred_1d)
    rmse = get_rmse(gt_1d_01, pred_1d_01)

    return kl, kl_01, cc, rmse, r2

def overlay_imp_on_img(img, imp, fname, colormap='jet'):

    cm = plt.get_cmap(colormap) # https://matplotlib.org/examples/color/colormaps_reference.html
    img2 = np.array(img, dtype=np.uint8)
    imp2 = np.array(imp, dtype=np.uint8)
    imp3 = (cm(imp2)[:, :, :3] * 255).astype(np.uint8)
    im_alpha = cv2.addWeighted(img2, 0.5, imp3, 0.5, 0)
    cv2.imwrite(fname, im_alpha)
