#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 15:00:15 2017
analysis_fmry.py
@author: jaime
"""

import os
import numpy as np
import pandas as pd
import warnings
import pdb
from datetime import datetime
import nitime.timeseries as ts
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
        label = ['PCC', 'lTPJ', 'rTPJ', 'mPFC']
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
    """return the labels of an atlas and coordinates 
    """
    from nilearn import datasets
    from nilearn.image import resample_img
    from collections import OrderedDict
    if label[0:4] == 'cort':
        dataset = datasets.fetch_atlas_harvard_oxford(label)
        labels = dataset.labels[1:]
        #Missing slices: Precentral Gyrus', (44, 51, 68))
        #Postcentral Gyrus', (52, 42, 69)
        # (73,59,32)->-68,6,-6
    
        dim_coords = OrderedDict([(labels[0],(48, 94,35)), (labels[1],(25,70 ,32 )),(labels[2],(12, 34, 56)),
	(labels[3],(25,72 ,55 )),(labels[4],(20, 77,40 )),(labels[5],(20, 71, 47)),(labels[6],(44,51 ,68 )),
	(labels[7],(61,69 ,17 )),(labels[8],(73, 59, 32)),(labels[9],(75, 52,35 )),(labels[10],(74, 59, 26)),
	(labels[11],(14, 51,30)),(labels[12],(72,35 ,38 )),(labels[13],(22, 61, 16)),(labels[14],(70,42 ,26 )),
	(labels[15],(70, 35,29 )),(labels[16],(52,42 ,69 )),(labels[17],(28, 39, 63)),(labels[18],(73, 48,53 )),
	(labels[19],(72,39 ,51 )),(labels[20],(21, 37, 55)),
	(labels[21],(60, 28, 57)),
	(labels[22],(68, 28, 38)),(labels[23],(47, 23, 40)),(labels[24],(45,84 ,27 )),(labels[25],(45, 63, 63)),
	(labels[26],(46,74 ,27 )),
	(labels[27],(45, 80, 52)),
	(labels[28],(45,61 ,57 )),(labels[29],(44,42 , 54)),(labels[30],(45,31 ,55 )),(labels[31],(45,23 ,49 )),
	(labels[32],(25,77 ,29 )),
	(labels[33],(57, 62,18 )),
	(labels[34],(58,45 , 28)),(labels[35],(41,22 ,33 )),(labels[36],(62,61 ,15 )),(labels[37],(64, 47, 54)),
	(labels[38],(62, 37, 29)),
	(labels[39],(33,23 ,30 )),
	(labels[40],(66, 74, 37)),(labels[41],(19,61 ,40 )),(labels[42],(17, 50, 47)),(labels[43],(20,60 ,36 )),
	(labels[44],(67, 52, 39)), (labels[45],(74, 53, 40)),(labels[46],(44, 21, 42)),(labels[47],(37,15 ,34 ))])  
        # MNI in mm
        dim_coords_mm = OrderedDict([(labels[0],(-18, 76,0)), (labels[1],(28,28 ,-6 )),(labels[2],(33, 73, 63)),
	(labels[3],(26,32 ,40 )),(labels[4],(38, 42,10 )),(labels[5],(38, 30, 24)),(labels[6],(-10,10 ,66 )),
	(labels[7],(-44,26 ,-6 )),(labels[8],(-68, 6, -6)),(labels[9],(-72, -8,0 )),(labels[10],(-70, 6, -18)),
	(labels[11],(50, -10,-10)),(labels[12],(-66,-42 ,6 )),(labels[13],(34, 10, -38)),(labels[14],(-62,-28 ,-18 )),
	(labels[15],(-62, -42,-12 )),(labels[16],(-26,-28 ,68 )),(labels[17],(22, -34, 56)),(labels[18],(-68, -16,36 )),
	(labels[19],(-66,-34 ,32 )),(labels[20],(36, -38, 40)),
	(labels[21],(-42, -56, 4)),
	(labels[22],(-58, -56, 6)),(labels[23],(-16, -66, 10)),(labels[24],(-12,56 ,-16 )),(labels[25],(-12, 14, 56)),
	(labels[26],(-14,36 ,-16 )),
	(labels[27],(-12, 48, 34)),
	(labels[28],(-12,10,44)),(labels[29],(-10,-28, 38)),(labels[30],(-12,50 ,40)),(labels[31],(-12,-66 ,28 )),
	(labels[32],(28,-38 ,-12)),
	(labels[33],(-36, 12,-34)),
	(labels[34],(-38,-22, -14)),(labels[35],(-4,-68,-4)),(labels[36],(-46,10 ,-40 )),(labels[37],(-50, -18, 38)),
	(labels[38],(-46, -38, -12)),
	(labels[39],(12,-66 ,-10 )),
	(labels[40],(-54, 36, 4)),(labels[41],(40,10 ,10 )),(labels[42],(44, -12, 24)),(labels[43],(38,8,2)),
	(labels[44],(-56, -8, 8)), (labels[45],(-70, -6, 10)),(labels[46],(-10, -70, 14)),(labels[47],(4,-82,-2))])                 
    else:
        #power 2011 map
        #dataset = datasets.fetch_coords_power_2011()
        #coords = np.vstack((dataset.rois['x'], dataset.rois['y'], dataset.rois['z'])).T
        #msdl  map
        dataset = datasets.fetch_atlas_msdl()
        # Loading atlas image stored in 'maps'
        atlas_filename = dataset['maps']
        # Loading atlas data stored in 'labels'
        labels = dataset['labels']
        coords = dataset.region_coords
    return dim_coords_mm
                
                          
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
        #dataset = datasets.fetch_coords_power_2011()
        atlas_filename = dataset.maps
        print('Atlas filename %s:',atlas_filename )
        masker = NiftiLabelsMasker(labels_img=atlas_filename, 
                                   smoothing_fwhm=preproc_parameters['smoothing_fwhm'], 
                                   standardize=preproc_parameters['standardize'],
                                   detrend=preproc_parameters['detrend'], 
                                   low_pass=preproc_parameters['low_pass'],
                                   high_pass=preproc_parameters['high_pass'],
                                   t_r=preproc_parameters['t_r'],verbose=5, memory_level=1)
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
        print('........Extracting image %d / %d', (i,len(epi_file)))
        ts = masker.fit_transform(epi_file[i])
        if ts.shape[0] == 120:
            warnings.warn("The time series number of points is 120, removing 4 initial dummy volumes", Warning)
            ts = ts[4:120]   
        time_series.append(ts)         
    print('Number of features:', len(time_series), 'Feature dimension:', time_series[0].shape)      
    #time_series[i].shape=(116, 4) (subjects x time x regions)
    return time_series

def build_granger_matrix(time_series, preproc_parameters=None, label_map=None):
    '''Calculate Granger causality for a set of time series using nitime.analysis.coherence 
    The Null hypothesis for grangercausalitytests is that the time series \
    in the second column, x2, does NOT Granger cause the time series in the first column, x1. \
    The null hypothesis for all four test is that the coefficients corresponding \
    to past values of the second time series are zero.
    '''
    
    import seaborn as sns
    from statsmodels.tsa.stattools import grangercausalitytests
    import matplotlib.pyplot as plt
    from nitime.analysis import  GrangerAnalyzer
    from nitime.viz import drawmatrix_channels
    #frequencies = np.linspace(0,0.2,129)
    
    # transpose to get time series (rois x time points)
    time_series = np.transpose(time_series)
    # change the type 
    time_series = ts.TimeSeries(time_series, sampling_interval=preproc_parameters['t_r'])
    order = 10
    granger = GrangerAnalyzer(time_series, order=order)
    listoffreqs_granger = granger.frequencies[:]
    freq_idx_granger = np.where((listoffreqs_granger > preproc_parameters['high_pass']) * (listoffreqs_granger < preproc_parameters['low_pass']))[0]
    corr_mats_xy = np.nan_to_num(np.mean(granger.causality_xy[:, :, freq_idx_granger], -1))
    corr_mats_yx = np.nan_to_num(np.mean(granger.causality_yx[:, :, freq_idx_granger], -1))
    #  must be significantly different than 0 for assuming correlation with a time-lag 
    corr_mats_diff = np.nan_to_num(np.mean(granger.causality_xy[:, :, freq_idx_granger] - granger.causality_yx[:, :, freq_idx_granger], -1))
    vmin = min(np.min(corr_mats_yx), np.min(corr_mats_xy))
    vmax= max(np.max(corr_mats_yx), np.max(corr_mats_xy))
    #plot the Granger correlation, Use a mask to plot only part of a matrix
    mask = np.zeros_like(corr_mats_xy)
    mask[np.triu_indices_from(mask)] = True
    # Three subplots sharing both x/y axes
    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    plt.setp(ax1.get_yticklabels(), rotation=45) 
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    plt.setp(ax3.get_xticklabels(), rotation=0) 
    msgtitle ="Granger causality (order=%d). x->y, y->x, x->y - y->x" % (order)
    ax1.set_title(msgtitle)
    #plot heatmap with xticks labels and values only jit it is not too big, eg DMN, avoid for Atlas
    if len(label_map) > 8:
        annot = False
        xticklabels = []
    else:
        annot = True
        xticklabels = label_map
    with sns.axes_style("white"):
        # Tell pointplot to plot on ax1 with the ax argument
        sns.heatmap(np.transpose(corr_mats_xy),xticklabels=xticklabels, mask=mask, vmax=vmax, vmin=vmin,center=0,cmap="YlGnBu", robust="True", annot=annot, ax=ax1)
        sns.heatmap(np.transpose(corr_mats_yx), xticklabels=xticklabels, mask=mask, vmax=vmax, vmin=vmin, center=0,cmap="YlGnBu", robust="True", annot=annot,ax=ax2)
        sns.heatmap(np.transpose(corr_mats_diff), xticklabels=xticklabels, mask=mask, center=0,cmap="YlGnBu", robust="True", annot=annot,ax=ax3)
        
    pdb.set_trace()
    return granger #[corr_mats_xy, corr_mats_yx, listoffreqs]
 
def test_for_granger(time_series, preproc_parameters=None, label_map=None, order=10):
    '''data for test whether the time series in the second column Granger causes the time series in the first column.
    four tests for granger non causality of 2 timeseries'''
    from statsmodels.tsa.stattools import grangercausalitytests
    from itertools import combinations
    def print_test_results(Gres_all, pairtestedlist=None):
        '''Plot the results of F and Chi test for Granger causality for all pairs and from 1 to order=order'''
        thr_v = 0.05
        print("Printing Granger test results")
        nb_of_tests = len(Gres_all)
        nb_of_orders_tested_for_pair = len(Gres_all[0])
        for testp in range(0,nb_of_tests):
            #print "Printing granger results for pair %d / %d." % (testp,nb_of_tests-1)
            for orderi in range(1,nb_of_orders_tested_for_pair+1):
                #accessing to the tuple Gres_all[0][1]
                #print " \n Calculating Granger test for timeseries pair:", pairtestedlist[orderi][0], '->', pairtestedlist[orderi][1]
                lr = Gres_all[testp][orderi][0].get('lrtest')[1]
                #print "pvalue %d %d %f" % (testp,orderi,lr)
                pf = Gres_all[testp][orderi][0].get('params_ftest')[1]
                ssrf = Gres_all[testp][orderi][0].get('ssr_ftest')[1]
                schi = Gres_all[testp][orderi][0].get('ssr_chi2test')[1]
                if ssrf <  thr_v:
                    print "Reject the Null Hypothesis A doesn't granger cause B for pair %d , order:%d lrtest=%f" % (testp,orderi,lr)
    
    if type(time_series) is np.ndarray:
        print "Time serties for only one subject"
        time_series = np.transpose(time_series)
    else:
        #pdb.set_trace()
        nb_of_subjects = len(time_series)
        print "Dimension of the time series number of subjects= %d:" % (nb_of_subjects)
        # mean across subjects (last dimension)
        time_series = np.transpose(time_series)
        time_series = time_series.mean(-1)
        
    df = pd.DataFrame(data=time_series, index=label_map)
    rois = df.shape[0]
    time_points = df.shape[1]
    print('Calculating granger causality for %d each with %d time points', rois, time_points)   
    pairs_list = combinations(label_map,2)
    print list(pairs_list)
    Gres_all = []
    pairtestedlist= []
    combinations = list(combinations(range(time_series.shape[0]),2))
    for co in combinations:
        pairofseries = [df.iloc[co[1]],df.iloc[co[0]]]
        pairofseries = np.asarray(pairofseries)
        pairofseries = np.transpose(pairofseries)
        pairtested = [label_map[co[0]], label_map[co[1]]]
        pairtestedlist.append(pairtested)
        print " \n\nCalculating Granger test for timeseries pair:", label_map[co[0]], '->', label_map[co[1]]
        grangerres = grangercausalitytests(pairofseries, maxlag=order, verbose=False)
        Gres_all.append(grangerres)  
        #pdb.set_trace()
        
    print_test_results(Gres_all, pairtestedlist)
    return Gres_all    

def build_sparse_invariance_matrix(time_series=None, label_map=None):
    '''calculates the sparse covariance and precision matrices matrix for a group of subjects
    time_series ndarray subjects x n x m and plots the connectome
    Input: subject_time_series ndarray'''
    from nilearn import plotting
    from nilearn.connectome import GroupSparseCovarianceCV
    edge_threshold = '90%'# 0.6 #'60%'
    print('Calling to nilearn.connectome.GroupSparseCovarianceCV \
        Sparse inverse covariance w/ cross-validated choice of the parameter')
    gsc = GroupSparseCovarianceCV(verbose=2)
    gsc.fit(time_series)
    pdb.set_trace()
    precision_matrix = -gsc.precisions_[...,0]
    covariances_matrix = gsc.covariances_[...,0]   
    plotconnectome  = True 
    if plotconnectome is True:
        plotting.plot_connectome(precision_matrix, label_map.values(), edge_threshold=edge_threshold,
                             title=str(edge_threshold)+'-GroupSparseCovariancePrec', display_mode='lzr')
        plotting.plot_connectome(covariances_matrix, label_map.values(), edge_threshold=edge_threshold,
                             title=str(edge_threshold)+'-GroupSparseCovariance', display_mode='lzr')
        #plot_covariance_matrix(gsc.covariances_[..., 0],gsc.precisions_[..., 0], labels, title = str(edge_threshold)+"-GroupSparseCovariance")
        plotting.show()
    # persistent homology analysis
    #persistent_homology(gsc.covariances_[..., 0], coords)
    #pdb.set_trace()

def build_seed_based_correlation_matrix(seed_masker, epi_file, time_series, non_seed_mask, preproc_parameters):
    ''' build_seed_based_correlation_matrix
    Input: seed_masker, time_series'''
    seed_time_series = time_series    
    brain_masker = generate_mask(non_seed_mask, epi_file, preproc_parameters)
    #generate non seed mask
    brain_time_series = brain_masker.fit_transform(epi_file)
    print("seed time series shape: (%s, %s)" % seed_time_series.shape)
    print("brain time series shape: (%s, %s)" % brain_time_series.shape)
    #make sure time series have same number of time points
    if seed_time_series.shape[0] != brain_time_series.shape:
        brain_time_series = brain_time_series[4:brain_time_series.shape[0],:]
        print("The corrected time series dimension are:")
        print("seed time series shape: (%s, %s)" % seed_time_series.shape)
        print("brain time series shape: (%s, %s)" % brain_time_series.shape)
    #select the seed
    seed_time_series = seed_time_series[:,0]
    seed_based_correlations = np.dot(brain_time_series.T, seed_time_series) / \
                          seed_time_series.shape[0]
    print "seed-based correlation shape", seed_based_correlations.shape
    print "seed-based correlation: min =", seed_based_correlations.min(), " max = ", seed_based_correlations.max()
    #Fisher-z transform the data to achieve a normal distribution. 
    #The transformed array can now have values more extreme than +/- 1.
    seed_based_correlations_fisher_z = np.arctanh(seed_based_correlations)
    print "seed-based correlation Fisher-z transformed: min =", seed_based_correlations_fisher_z.min(), \
    " max =", seed_based_correlations_fisher_z.max()                                                                                             
    seed_based_correlation_img = brain_masker.inverse_transform(seed_based_correlations.T)
    seed_based_correlation_img.to_filename('sbc_z.nii.gz')
    pcc_coords = dim_coords.values()[0]
    #pcc_coords = [(0, -52, 18)]
    #MNI152Template = '/usr/local/fsl/data/standard/MNI152_T1_2mm_brain_symmetric.nii.gz'
    #remove out-of brain functional connectivity using a mask
    icbms = datasets.fetch_icbm152_2009()
    masker_mni = NiftiMasker(mask_img=icbms.mask)
    data = masker_mni.fit_transform('sbc_z.nii.gz')
    masked_sbc_z_img = masker_mni.inverse_transform(data)
    #pdb_set_tarce()
    display = plotting.plot_stat_map(masked_sbc_z_img , cut_coords=pcc_coords, \
                                         threshold=0.6, title= 'PCC-based corr. V-A', dim='auto', display_mode='ortho')
    
    
    
    
    
    
    
    
    
    
    
    
    

def build_correlation_matrix(time_series, kind_of_analysis='time', kind_of_correlation='correlation'):
    ''' calculate the correlation matrix for the time series according to kind_of_analysis and
    kind_of_correlation:{“correlation”, “partial correlation”, “tangent”, “covariance”, “precision”} '''
    from nilearn.connectome import ConnectivityMeasure 
    from sklearn.covariance import LedoitWolf, EmpiricalCovariance
    # LedoitWolf estimatorhas slightly shrunk towards zero compared to a maximum-likelihood estimate
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
    from nilearn import plotting
    from nitime.viz import drawmatrix_channels, drawgraph_channels
    # plot heatmap and network using nitime (no brain in background)
    plot_heatmap = True
    plot_graph = True
    if plot_heatmap == True:
        print('Plotting correlation_matrix as a heatmap from nitime...')
        fig_h_drawx = drawmatrix_channels(corr_matrix, label_map.keys(), size=[10., 10.], color_anchor=0, title= msgtitle)    
    if plot_graph == True:    
        print('Plotting correlation_matrix as a network nitime...')
        fig_g_drawg = drawgraph_channels(corr_matrix, label_map.keys(),title=msgtitle)
    #plotting connectivity network with brain overimposed
    plotting.plot_connectome(corr_matrix, label_map.values(),edge_threshold='90%', title=msgtitle,display_mode="ortho",edge_vmax=.5, edge_vmin=-.5)       