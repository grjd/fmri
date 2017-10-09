#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 15:00:15 2017
analysis_fmry.py
@author: jaime
"""

import os
import pdb
from datetime import datetime
from nilearn.image import mean_img, index_img
from nilearn.plotting import (plot_roi, plot_epi,plot_prob_atlas, find_xyz_cut_coords, show,
                              plot_stat_map)    
    
def group_ICA(epi_file_list, preproc_parameters, cohort=None):
    """Decomposition analysis of bold data, using two estimators: ICA and dictLearning. 
    The estimators calculate the components of a list of bold images
    input: epi_file_list list of images,preproc_parameters,cohort 
        
    """
    from nilearn.decomposition import DictLearning,CanICA
    #from nilearn.plotting import plot_prob_atlas
    from nilearn.image import iter_img
    #from nilearn.plotting import plot_stat_map, show
    #Parameters of the estimator call
    n_components = 12
    verbose = 5
    random_state = 0
    memory_level = 2
    memory="nilearn_cache"
    dirname = os.path.dirname(epi_file_list[0])
    # Define 2 estimators. Dictionary learning and CanICA
    print('Analysing the group DictLearning for list of images:', epi_file_list)
    dict_learning = DictLearning(n_components=n_components, 
                                standardize=preproc_parameters['standardize'], 
                                smoothing_fwhm=preproc_parameters['smoothing_fwhm'], 
                                detrend=preproc_parameters['detrend'],
                                low_pass=preproc_parameters['low_pass'], 
                                high_pass=preproc_parameters['high_pass'],
                                t_r=preproc_parameters['t_r'],
                                memory=memory, memory_level=memory_level,
                                verbose=verbose,random_state=random_state,n_epochs=1)     
    # CanICA
    print('Analysing the group ICA for list of images:', epi_file_list)
    canica = CanICA(n_components=n_components,
                    standardize=preproc_parameters['standardize'], 
                    smoothing_fwhm=preproc_parameters['smoothing_fwhm'], 
                    detrend=preproc_parameters['detrend'],
                    low_pass=preproc_parameters['low_pass'], 
                    high_pass=preproc_parameters['high_pass'],
                    t_r=preproc_parameters['t_r'],
                    memory=memory, memory_level=memory_level, threshold=3.,
                    verbose=verbose, random_state=random_state)    
    # Fit both estimators
    estimators = [dict_learning, canica]
    names = {dict_learning: 'DictionaryLearning', canica: 'CanICA'}
    
    components_imgs = []
    for estimator in estimators:
        print('[Example] Learning maps using %s model' % names[estimator])
        estimator.fit(epi_file_list)
        print('[Example] Saving results')
        # Decomposition estimator embeds their own masker
        masker = estimator.masker_
        # Save output maps to a Nifti   file
        components_img = masker.inverse_transform(estimator.components_)
        filenameres = "{}_resting_state.nii.gz".format(names[estimator])
        imageresult = os.path.join(dirname, filenameres)
        components_img.to_filename(imageresult)
        components_imgs.append(components_img)
    # Visualize the results
    # Selecting specific maps (components) to display
    # we have 0..n_components_1 for each estimator
    indices = {dict_learning: 0, canica: 0}
    # We select relevant cut coordinates for displaying
    cut_component = index_img(components_imgs[0], indices[dict_learning])
    cut_coords = find_xyz_cut_coords(cut_component)
    #pdb.set_trace()
    for estimator, components in zip(estimators, components_imgs):
        # 4D plotting
        plot_prob_atlas(components, view_type="filled_contours",
                    title="%s components %s" % (names[estimator], cohort),
                    cut_coords=cut_coords, colorbar=False)
        # plot the map for each ICA component separately
        plot_all_components = True
        if plot_all_components is True:
            print('Plotting each component separately')
            for i, cur_img in enumerate(iter_img(components)):
                plot_stat_map(cur_img, title="%s IC %d" % (names[estimator], i), cut_coords=cut_coords, colorbar=False)
                #plot only z axis
                #plot_stat_map(cur_img, display_mode="z", title="%s IC %d" % (names[estimator], i), cut_coords=1, colorbar=False)
        else:
            print('Plotting one component: %s', format(indices[estimator]))
            plot_stat_map(index_img(components, indices[estimator]),
                  title="%s component:%s %s" % (names[estimator], format(indices[estimator]),cohort),
                  cut_coords=cut_coords, colorbar=False)
    show()
       
def clustering_Ward(epi_file_list, preproc_parameters, cohort=None):
    """Computes Ward type Hierarchical clustering Ward
    """
    from sklearn.cluster import FeatureAgglomeration
    from sklearn.feature_extraction import image
    #import time
    from nilearn import input_data
    
    nifti_masker = input_data.NiftiMasker(smoothing_fwhm=preproc_parameters['smoothing_fwhm'], 
                                          standardize=preproc_parameters['standardize'], 
                                          detrend=preproc_parameters['detrend'],
                                          low_pass=preproc_parameters['low_pass'], 
                                          high_pass=preproc_parameters['high_pass'], 
                                          t_r=preproc_parameters['t_r'],
                                          memory='nilearn_cache',
                                          mask_strategy='epi', verbose=5, memory_level=3)  
    print('compute the mask and extracts the time series form the file(s)')
    fmri_list_m = []
    fmri_masked = nifti_masker.fit_transform(epi_file_list[0])
    fmri_list_m.append(fmri_masked)
    fmri_masked = nifti_masker.fit_transform(epi_file_list[0])
    fmri_list_m.append(fmri_masked)
    #fmri_masked = nifti_masker.fit(epi_file_list)
    
    print('retrieve the nup array of the mask')
    mask = nifti_masker.mask_img_.get_data().astype(bool)
    print('compute the connectivity matrix for the mask')
    shape = mask.shape
    connectivity = image.grid_to_graph(n_x=shape[0], n_y=shape[1],
                                   n_z=shape[2], mask=mask)
    #pdb.set_trace()
    tic = datetime.now()
    #FeatureAgglomeration clustering algorithm from scikit-learn 
    n_clusters= 40
    print('computing Ward...')
    ward = FeatureAgglomeration(n_clusters, connectivity=connectivity,
                            linkage='ward', memory='nilearn_cache')
    pdb.set_trace()
    #ward.fit(fmri_masked)
    ward.fit(fmri_list_m)
    print("Ward agglomeration %d clusters: " % n_clusters, "in time=", str(datetime.now()- tic))
    #visualize the results
    labels = ward.labels_ + 1
    labels_img = nifti_masker.inverse_transform(labels)
    mean_func_img = mean_img(epi_file_list)
    msgtitle ="Ward parcellation nclusters=%s, %s" % (n_clusters, os.path.split(os.path.dirname(epi_file_list))[1])
    first_plot = plot_roi(labels_img, mean_func_img, title=msgtitle,
                      display_mode='ortho', cut_coords=(0,-52,18))
    cut_coords = first_plot.cut_coords
    print ('cut coords:', cut_coords)
    dirname = os.path.dirname(epi_file_list[0])
    imageresult = os.path.join(dirname, 'ward_parcellation.nii.gz')
    print('Result is the file ward_parcellation.nii.gz in', dirname) 
    labels_img.to_filename(imageresult)