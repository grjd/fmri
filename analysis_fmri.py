#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 15:00:15 2017
analysis_fmry.py
@author: jaime
"""

import os
import numpy as np
import warnings
import pdb
from datetime import datetime
from nilearn.image import mean_img, index_img, image
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
    """Computes Ward type Hierarchical unsupervised learning algorithm (Ward)
    """
    from sklearn.cluster import FeatureAgglomeration
    from sklearn.feature_extraction import image
    #import time
    from nilearn import input_data
    from nilearn.image import concat_imgs
    
    nifti_masker = input_data.NiftiMasker(smoothing_fwhm=preproc_parameters['smoothing_fwhm'], 
                                          standardize=preproc_parameters['standardize'], 
                                          detrend=preproc_parameters['detrend'],
                                          low_pass=preproc_parameters['low_pass'], 
                                          high_pass=preproc_parameters['high_pass'], 
                                          t_r=preproc_parameters['t_r'],
                                          memory='nilearn_cache',
                                          mask_strategy='epi', verbose=5, memory_level=3) 
    #nifti_masker = input_data.NiftiLabelsMasker(labels_img=atlas_filename,
    #                                            standardize=True, memory='nilearn_cache', verbose=5) 
    
    #concatenate list of 4D images into a single 4D image 
    pdb.set_trace()
    conc_epi_file_list = concat_imgs(epi_file_list)
    print('compute the mask and extracts the time series form the file(s)')
    fmri_masked = nifti_masker.fit_transform(conc_epi_file_list)
    
    mask = nifti_masker.mask_img_.get_data().astype(bool)
    shape = mask.shape
    print('compute the connectivity matrix for the mask', shape)
    #connectivity =<204867x204867 sparse matrix of type '<type 'numpy.int64'>'
    #connectivity matrix: which voxel is connected to which
    connectivity = image.grid_to_graph(n_x=shape[0], n_y=shape[1],
                                   n_z=shape[2], mask=mask)
    #pdb.set_trace()
    tic = datetime.now()
    #FeatureAgglomeration clustering algorithm from scikit-learn 
    n_clusters= 40
    print('computing Ward...')
    #FeatureAgglomeration(affinity='euclidean', compute_full_tree='auto',connectivity=<204867x204867 sparse matrix of type '<type 'numpy.int64'>'with 1405595 stored elements in COOrdinate format>,linkage='ward', memory='nilearn_cache', n_clusters=40,pooling_func=<function mean at 0x1024aaa28>)
    #linkage : {“ward”, “complete”, “average”}, optional, default “ward”:ward minimizes the variance of the clusters being merged.
    #Ward only accepts  affinity='euclidean'. 
    ward = FeatureAgglomeration(n_clusters=n_clusters, connectivity=connectivity,
                            linkage='ward', memory='nilearn_cache')
    #pdb.set_trace()
    ward.fit(fmri_masked)
    print("Ward agglomeration %d clusters: " % n_clusters, "in time=", str(datetime.now()- tic))
    #visualize the results
    labels = ward.labels_ + 1
    labels_img = nifti_masker.inverse_transform(labels)
    #mean_img(epi_file_list_conc) provides same result
    mean_func_img = mean_img(epi_file_list)
    #msgtitle ="Ward parcellation nclusters=%s, %s" % (n_clusters, os.path.split(os.path.dirname(epi_file_list))[1])
    msgtitle ="Ward parcellation:%s, nclusters=%s" % (cohort,n_clusters) 
    first_plot = plot_roi(labels_img, mean_func_img, title=msgtitle,
                      display_mode='ortho', cut_coords=(0,-52,18))
    cut_coords = first_plot.cut_coords
    print ('cut coords:', cut_coords)
    dirname = os.path.dirname(epi_file_list[0])
    filenameres = "ward_parcellation.nii.gz_{}.nii.gz".format(cohort)
    imageresult = os.path.join(dirname, filenameres)
    print('Result is saved in file %s', imageresult) 
    labels_img.to_filename(imageresult)
    
def get_MNI_coordinates(label):
    '''get_MNI_coordinates returns a OrderedDict of coordinates of the label in MNI space
    Input: label =DMN(default mode network), AN(attention network), SN(salience network), random(random coordinates)
    Example: get_MNI_coordinates('DMN')
    '''
    from collections import OrderedDict
    from random import randint
    print('Calling to get_MNI_coordinates for label:%s', label)
    max_coords_MNI =[(-78,78),(-112,76),(-70,86)]
    dim_coords = []
    if label is 'DMN':
        # http://sprout022.sprout.yale.edu/mni2tal/mni2tal.html
        # DMN coordinates from HEDDEN ET AL (2009) PCC
        # DMN = PCC (-5, -53, 41) is BA31 http://www.sciencedirect.com/science/article/pii/S187892931400053X
        # MPFC (0, 52, -6)  LLPC (-48, -62, 36) RLPC (46, -62, 32)
        label = ['Post. Cing. Cortex','Left Tmp.Ptl. junction','Right Tmp.Ptl. junction','Medial PFC '] 
        # http://nilearn.github.io/auto_examples/03_connectivity/plot_adhd_spheres.html#sphx-glr-auto-examples-03-connectivity-plot-adhd-spheres-py
        dim_coords = OrderedDict([(label[0],(0, -52, 18)),(label[1],(-46, -68, 32)), (label[2],(46, -68, 32)),(label[3],(1, 50, -5))])
        #make sure dictionary respects the order of the keys
    elif label is 'SN':
        # Salience network
        dim_coords = []    
    elif label is 'AN':
        #Attention network
        dim_coords=[]
    elif label is 'random':
        #build a network of numnodes randomly generated
        numofnodes = 1
        for i in range(0,numofnodes):
            dim_coord=[(randint(max_coords_MNI[0][0], max_coords_MNI[0][1])),
                    (randint(max_coords_MNI[1][0], max_coords_MNI[1][1])),
                    (randint(max_coords_MNI[2][0], max_coords_MNI[2][1]))]
            dim_coords.append(dim_coord)
        pdb.set_trace()
    else: 
        print " ERROR: label:", label, " do not found, returning empty list of coordinates!"   
    return dim_coords   

def get_atlas_labels(label):
    """return the labels of an atlas
    """
    from nilearn import datasets
    if label[0:4] == 'cort':
        dataset = datasets.fetch_atlas_harvard_oxford(label)
        legend_values = dataset.labels[1:]
        return legend_values
    
def generate_mask(mask_type, mask_coords, preproc_parameters):
    '''generate_mask returns a mask object depending on the mask type
    '''
    from nilearn import datasets
    from nilearn.input_data import NiftiLabelsMasker, NiftiSpheresMasker, NiftiMasker, NiftiMapsMasker 
    masker = []
    #atlas type of mask
    if mask_type == 'atlas':
        atlas_name = mask_coords
        print("Loading Harvard-Oxford parcellation from FSL if installed.\
        If not, it downloads it and stores it in NILEARN_DATA directory. \
        Atlas to load: cort-maxprob-thr0-1mm, cort-maxprob-thr0-2mm, \
        cort-maxprob-thr25-1mm, cort-maxprob-thr25-2mm, cort-maxprob-thr50-1mm, \
        cort-maxprob-thr50-2mm, sub-maxprob-thr0-1mm, sub-maxprob-thr0-2mm, \
        sub-maxprob-thr25-1mm, sub-maxprob-thr25-2mm, sub-maxprob-thr50-1mm, \
        sub-maxprob-thr50-2mm, cort-prob-1mm, cort-prob-2mm, sub-prob-1mm, sub-prob-2mm .")
        print('Selected atlas to download is %s', atlas_name)
        dataset = datasets.fetch_atlas_harvard_oxford(atlas_name)
        atlas_filename = dataset.maps
        print('Atlas filename %s:',atlas_filename )
        masker = NiftiLabelsMasker(labels_img=atlas_filename, 
                                   smoothing_fwhm=preproc_parameters['smoothing_fwhm'], 
                                   standardize=preproc_parameters['standardize'],
                                   detrend=preproc_parameters['detrend'], 
                                   low_pass=preproc_parameters['low_pass'],
                                   high_pass=preproc_parameters['high_pass'],
                                   t_r=preproc_parameters['t_r'],verbose=5, memory_level=3)
        plotting_atlas = True
        if plotting_atlas is True:
            msgtitle = "Selected atlas:{}".format(atlas_name)
            plot_roi(atlas_filename, title=msgtitle)
    elif mask_type == 'DMN':
        # Extract the coordinates from the dictionary
        dim_coords = mask_coords
        print " The mask is the list of voxels:", dim_coords.keys(), "in MNI space:", dim_coords.values()
        masker = NiftiSpheresMasker(dim_coords.values(), radius=8,
                                               detrend=preproc_parameters['detrend'],
                                               smoothing_fwhm=preproc_parameters['smoothing_fwhm'],
                                               standardize=preproc_parameters['standardize'],
                                               low_pass=preproc_parameters['low_pass'],
                                               high_pass=preproc_parameters['high_pass'], 
                                               t_r=preproc_parameters['t_r'],memory='nilearn_cache', 
                                               memory_level=3, verbose=5, allow_overlap=False)         
    return masker  

def extract_timeseries_from_mask(masker, epi_file):
    ''' extract time series from mask object'''
    time_series = []
    for i in range(0, len(epi_file)):
        print('........Extracting image %d / %d', (i,len(epi_file) ))
        ts = masker.fit_transform(epi_file[i])
        if ts.shape[0] == 120:
            warnings.warn("The time series number of points is 120, removing 4 initial dummy volumes", Warning)
            ts = ts[4:120]   
        time_series.append(ts)         
    print('Number of features:', len(time_series), 'Feature dimension:', time_series[0].shape)      
    #time_series[i].shape=(116, 4) (subjects x time x regions)
    return time_series

def build_correlation_matrix(time_series, kind_of_analysis='time', kind_of_correlation='correlation'):
    ''' calculate the correlation matrix for the time series according to kind_of_analysis and
    kind_of_correlation:{“correlation”, “partial correlation”, “tangent”, “covariance”, “precision”} '''
    from nilearn.connectome import ConnectivityMeasure 
    from sklearn.covariance import LedoitWolf, EmpiricalCovariance
    
    connectivity_measure = ConnectivityMeasure(EmpiricalCovariance(assume_centered=True), kind=kind_of_correlation) 
    connectivity_measure = ConnectivityMeasure(LedoitWolf(assume_centered=True), kind=kind_of_correlation) 
    correlation_matrices = []
    for i in range(0, len(time_series)):
        correlation_matrix = connectivity_measure.fit_transform([time_series[i]])[0]
        correlation_matrices.append(correlation_matrix)
    print('Built correlation matrices:%s, dimension %d x%d x%d', kind_of_correlation, len(correlation_matrices), correlation_matrices[0].shape[0],correlation_matrices[0].shape[1] )    
    arr_corr_matrices = np.array(correlation_matrices)
    print('The wise element mean of the correlation matrices is:')
    wisemean = arr_corr_matrices.mean(axis=0)
    print(wisemean)
    print('The overall mean of the correlation matrix for each subject is:  ')
    means = [np.mean([el for el in sublist]) for sublist in correlation_matrices]
    print(means)
    arrmeans = np.array(means)
    print('The mean across subjects is %.3f and the std is %.3f', np.mean(arrmeans), np.std(arrmeans))    
    return correlation_matrices

def plot_correlation_matrix(corr_matrix,label_map,msgtitle=None):
    ''' plot correlation matrix
    Input: ONE correlation matrix
    label_map : list of rois
    masgtitle'''

    from nitime.viz import drawmatrix_channels, drawgraph_channels
    plot_heatmap = True
    plot_graph = True
    if plot_heatmap == True:
        print('Plotting correlation_matrix as a heatmap from nitime...')
        fig_h_drawx = drawmatrix_channels(corr_matrix, label_map, size=[10., 10.], color_anchor=0, title= msgtitle)    
    if plot_graph == True:    
        print('Plotting correlation_matrix as a network nitime...')
        fig_g_drawg = drawgraph_channels(corr_matrix, label_map,title=msgtitle)
        
    

        #plotting.plot_connectome(correlation_matrix, legend_values,edge_threshold='05%', title=msgtitle,display_mode="ortho",edge_vmax=.5, edge_vmin=-.5)       