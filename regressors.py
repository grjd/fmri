#https://gist.github.com/KamalakerDadi/af531d09eb47eb98cd82272cda6704c0
#You can use segmented T1 images as a "mask" to extract signals on fmri images.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nilearn import masking
from nilearn import signal, _utils
from nilearn.image import high_variance_confounds, resample_img, new_img_like, math_img, resample_to_img
#from nilearn._utils.compat import get_affine
from nilearn._utils.niimg_conversions import _check_same_fov
import collections
import os
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
        

def extract_timeries_with_confounds(func_data, f_confounds):
    """ extract_timeries_with_confounds: TEST Extract time series with confounds file.
    compare differences using confounds and not using
    Args: f_confounds csv file with confounds
    """
    # Loading atlas image stored in 'maps'
    from nilearn import datasets
    import analysis_fmri as afmri

    atlas = datasets.fetch_atlas_msdl()
    atlas_filename = atlas['maps']
    atlas_labels = atlas['labels']
    data = datasets.fetch_adhd(n_subjects=1)
    print('First subject resting-state nifti image (4D) is located at: %s' % func_data)
    # Extract time series
    from nilearn.input_data import NiftiMapsMasker
    masker = NiftiMapsMasker(maps_img=atlas_filename, standardize=True, memory='/tmp/nilearn_cache', verbose=5)
    time_series = masker.fit_transform(func_data)
    
    time_series_con = masker.fit_transform(func_data, confounds= f_confounds)
    # plot the time series difference
    fig = plt.figure(figsize= (6,9))
    ax0 = fig.add_subplot(121)
    ax1 = fig.add_subplot(122)
    ax0.imshow(time_series, interpolation='none', origin='lower')
    ax1.imshow(time_series_con, interpolation='none', origin='lower')
    ax0.set_title(r"""time series""")
    ax1.set_title(r"""time series con""")
    time_series_clean = afmri.extract_timeseries_from_mask(masker, func_data)
    time_series_clean_con = afmri.extract_timeseries_from_mask(masker, func_data, f_confounds)
    fig = plt.figure(figsize= (6,9))
    ax0 = fig.add_subplot(121)
    ax1 = fig.add_subplot(122)
    ax0.imshow(time_series_clean, interpolation='none', origin='lower')
    ax1.imshow(time_series_clean_con, interpolation='none', origin='lower')
    ax0.set_title(r"""time series clean""")
    ax1.set_title(r"""time series clean con""")
    print('END')


def main(func=None,mask_img_white=None, mask_img_csf=None, mcf_confounds=None):
    print('Generate the confounding matrix for white and csf....\n')
    #Select the fmri image in MNI space, must have w preffix
    func = "/Users/jaime/vallecas/data/parpadeo/images_0/Subject_0742_y7/w__fMRI_RESTING_S_20180510111831_10_mcf.nii.gz"
    image_directory = os.path.dirname(func)
    #select mask image    : CSF _0, Gray_1, White_2
    if mask_img_white is None: mask_img_white = "/Users/jaime/vallecas/data/parpadeo/images_0/Subject_0742_y7/__SAG_3D_IR_20180510111831_3.anat/T1_fast_pve_2.nii.gz"
    if mask_img_csf is None: mask_img_csf = "/Users/jaime/vallecas/data/parpadeo/images_0/Subject_0742_y7/__SAG_3D_IR_20180510111831_3.anat/T1_fast_pve_0.nii.gz"
    mask_confounds = dict({'white':mask_img_white, 'csf':mask_img_csf})
    df = pd.DataFrame({'white': [], 'csf': []})
    print('Calling to compute_confounds for {0} factors \n'.format(mask_confounds.keys()))
    for key, value in mask_confounds.iteritems():
        print('Generating confounding for mask type:{0} ...\n'.format(key))
        confounds = compute_confounds(func, value, 1)
        print(' \n DONE with compute_confounds={0} .\n'.format(key))
        df[key]= pd.Series(confounds)
    
    print('\n *** compute_confounds FINISHED for Masks: {0} *** \n'.format(mask_confounds))
    df_wc = pd.DataFrame(df)
    print('\n Getting the mcf ouliers to convert into pandas Series ...\n')
    #Get the mcf ouliers and convert into pandas Series
    mcf_confounds = "/Users/jaime/vallecas/data/parpadeo/images_0/Subject_0742_y7/mcf_results/__fMRI_RESTING_S_20180510111831_10_mcf.nii.gz.par"
    df_mcf = pd.read_csv(mcf_confounds, float_precision='high',delim_whitespace=True, names = ('p1','p2','p3','p4','p5','p6'))
    print('\n *** Motion correction confounds FINISHED. \n\n')
    print('\n Consolidating motion and white-csf confounds in one array.... \n')
    frames = [df_mcf, df_wc]
    df_all_confounds = pd.concat(frames, axis=1, join='outer', join_axes=None, keys=None, verify_integrity=True, copy=True)
    print('CREATED confounds.csv:: p1      p2      p3      p4      p5      p6      csf     white \n')
    # Save confounds file
    f_confounds = os.path.join(image_directory,'confounds.csv')
    df_all_confounds.to_csv(f_confounds, sep='\t', encoding='utf-8', header=False)
    print('Saved confounds file at: %s/%s \n' % (os.getcwd(), f_confounds))
    # TEST confounds: extract time series with and without confounds
    extract_timeries_with_confounds(func, f_confounds)    

    










    
