#https://gist.github.com/KamalakerDadi/af531d09eb47eb98cd82272cda6704c0
#ou can use segmented T1 images as a "mask" to extract signals on fmri images.

import numpy as np
import pandas as pd
from nilearn import masking
from nilearn import signal, _utils
from nilearn.image import high_variance_confounds, resample_img, new_img_like, math_img, resample_to_img
#from nilearn._utils.compat import get_affine
from nilearn._utils.niimg_conversions import _check_same_fov
import collections
import pdb


def compute_confounds(imgs, mask_img, n_confounds=5, get_randomized_svd=False,
                      compute_not_mask=False):
    """imgs 4D images, compute_not_mask True for non mask confound extraction
    """
    confounds = []
    if not isinstance(imgs, collections.Iterable) or \
            isinstance(imgs, basestring):
        imgs = [imgs, ]

    img = _utils.check_niimg_4d(imgs[0])
    print(img.header)
    #type(img) is nibabel.nifti1.Nifti1Image
    #shape of first 3 dimensions without time
    shape = img.shape[:3]
    #get_affine for old versions of nilearn
    #affine = get_affine(img)
    affine = img.affine

    #mask_img cant be a list but an instance , a 3D nii
    if isinstance(mask_img, basestring):
        mask_img = _utils.check_niimg_3d(mask_img)
        data = mask_img.get_data(caching='unchanged')
    mask_img_resampled = math_img('img >.5', img=resample_to_img(mask_img, img))
    if not _check_same_fov(img, mask_img):
        print('Resampling the image to have same fov...\n')
        mask_img2 = resample_img(
            mask_img, target_shape=shape, target_affine=affine,
            interpolation='nearest')
        data2 = mask_img2.get_data(caching='unchanged')
    #pdb.set_trace()    
    if compute_not_mask:
        print("Non mask based confounds extraction")
        not_mask_data = np.logical_not(mask_img.get_data().astype(np.int))
        whole_brain_mask = masking.compute_multi_epi_mask(imgs)
        not_mask = np.logical_and(not_mask_data, whole_brain_mask.get_data())
        mask_img = new_img_like(img, not_mask.astype(np.int), affine)
    #pdb.set_trace()    
    for img in imgs:
        print("[Confounds Extraction] {0}".format(img))
        img = _utils.check_niimg_4d(img)
        print("[Confounds Extraction] high Variance confounds computation]")
        
        high_variance = high_variance_confounds(img, mask_img=mask_img_resampled, n_confounds=n_confounds)
        if compute_not_mask and get_randomized_svd:
            signals = masking.apply_mask(img, mask_img)
            non_constant = np.any(np.diff(signals, axis=0) != 0, axis=0)
            signals = signals[:, non_constant]
            signals = signal.clean(signals, detrend=True)
            print("[Confounds Extraction] Randomized SVD computation")
            U, s, V = randomized_svd(signals, n_components=n_confounds,
                                     random_state=0)
            if high_variance is not None:
                confound_ = np.hstack((U, high_variance))
            else:
                confound_ = U
        else:
            confound_ = high_variance
        confounds.append(confound_)
    confounds = np.array(confounds)
    return confounds.reshape(confounds.shape[1]) 
        
    #return np.array(confounds)
    #return confounds

def main(func=None,mask_img_white=None, mask_img_csf=None):
    print('Generate the confounding matrix for white and csf....\n')
    #Select the fmri image in MNI space, must have w preffix
    func = "/Users/jaime/vallecas/data/uma/Subject_UMA_0018/wbold_data.nii.gz"
    #select mask image    
    if mask_img_white is None: mask_img_white = "/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_control/T1s/0000_T1.anat/T1_fast_pve_0_MNI.nii.gz"
    if mask_img_csf is None: mask_img_csf = "/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_control/T1s/0000_T1.anat/T1_fast_pve_2_MNI.nii.gz"
    mask_confounds = dict({'white':mask_img_white, 'csf':mask_img_csf})
    df = pd.DataFrame({'white': [], 'csf': []})
    print('Calling to compute_confounds for {0} factors \n'.format(mask_confounds.keys()))
    for key, value in mask_confounds.iteritems():
        print('Generating confounding for mask type:{0} ...\n'.format(key))
        confounds = compute_confounds(func, value, 1)
        print(' \n DONE with compute_confounds={0} ....\n'.format(key))
        df[key]= pd.Series(confounds)
    
    print('\n *** compute_confounds FINISHED for Masks: {0} *** \n'.format(mask_confounds))
    df_wc = pd.DataFrame(df)
    #Get the mcf ouliers and convert into pandas Series
    mcf_confounds = "/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_control/mcf_results/0859_outliers.txt"
    df_mcf = pd.read_csv(mcf_confounds)
    df_all_confounds = pd.concat([df_wc,df_mcf], axis=1, ignore_index=True)
    df_all_confounds.fillna(0, inplace=True)
    file_name = 'counfounds.csv'
    df_all_confounds.to_csv(file_name, sep='\t', encoding='utf-8')
    