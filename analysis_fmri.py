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
from scipy import stats, signal, linalg
import matplotlib.pyplot as plt
  
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
                msgoutputfile = 'canICA_' + 'i:'+ str(i) + '.png'
                msgoutputfile = 'figures/' + msgoutputfile
                plot_stat_map(cur_img, title="%s IC %d" % (names[estimator], i), output_file = msgoutputfile, cut_coords=cut_coords, colorbar=False)
                #plot only z axis
                #plot_stat_map(cur_img, display_mode="z", title="%s IC %d" % (names[estimator], i), cut_coords=1, colorbar=False)
        else:
            print('Plotting one component: %s', format(indices[estimator]))
            msgoutputfile = 'canICA_' + 'ALL' + '.png'
            msgoutputfile = 'figures/' + msgoutputfile
            plot_stat_map(index_img(components, indices[estimator]), output_file = msgoutputfile, 
                  title="%s component:%s %s" % (names[estimator], format(indices[estimator]),cohort),
                  cut_coords=cut_coords, colorbar=False)
    #show()
       
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
    conc_epi_file_list = concat_imgs(epi_file_list)
    print('Compute the mask and extracts the time series from the file(s) \n')
    fmri_masked = nifti_masker.fit_transform(conc_epi_file_list)
    
    mask = nifti_masker.mask_img_.get_data().astype(bool)
    shape = mask.shape
    print('Compute the connectivity matrix for the mask \n', shape)
    #connectivity =<204867x204867 sparse matrix of type '<type 'numpy.int64'>'
    #connectivity matrix: which voxel is connected to which
    connectivity = image.grid_to_graph(n_x=shape[0], n_y=shape[1],
                                   n_z=shape[2], mask=mask)
    tic = datetime.now()
    #FeatureAgglomeration clustering algorithm from scikit-learn 
    n_clusters= 40
    print('Computing Ward... \n')
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
    msgoutputfile = 'Ward_' + 'nclusters_'+ str(n_clusters) + '.png'
    msgoutputfile = 'figures/' + msgoutputfile
    cut_coords = (0,-52,18)
    first_plot = plot_roi(labels_img, mean_func_img, title=msgtitle, output_file= msgoutputfile, display_mode='ortho', cut_coords=cut_coords)
    #print ('cut coords:', cut_coords)
    dirname = os.path.dirname(epi_file_list[0])
    #save the nift image
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

    else: 
        print " ERROR: label:", label, " do not found, returning empty list of coordinates!"   
    print("MNI coordinates :{}",dim_coords)
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
        print('MSDL atlas!!')
        atlas = datasets.fetch_atlas_msdl()
        # Loading atlas image stored in 'maps'
        atlas_filename = atlas['maps']
        # Loading atlas data stored in 'labels'
        labels = atlas['labels']
        dim_coords_mm = atlas.region_coords

    return dim_coords_mm
                
                          
def generate_mask(mask_type, mask_coords, preproc_parameters, epi_filename=None):
    '''generate_mask returns a mask object depending on the mask type
    '''
    from nilearn import datasets
    from nilearn.input_data import NiftiLabelsMasker, NiftiSpheresMasker, NiftiMasker, NiftiMapsMasker 
    from nilearn.plotting import plot_roi, show
    from nilearn.datasets import load_mni152_template
    from nilearn.image import resample_to_img
    template = load_mni152_template()
    from nilearn.image import smooth_img
    masker = []
    print('Generatimg mask....')

    #atlas type of mask
    if mask_type == 'atlas':
        plotting_atlas = True
        if mask_coords[0:4] == 'cort':
            plotting_atlas_probabilistic = False
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
        elif mask_coords[0:4] == 'msdl':
            plotting_atlas_probabilistic = True
            plotting_atlas = False
            atlas_name = mask_coords
            print('Loading Probabilistic MSDL Atlas:{} \n', atlas_name)
            atlas = datasets.fetch_atlas_msdl()
            atlas_filename = atlas['maps']
            labels = atlas['labels']
            coords = atlas.region_coords
            masker = NiftiLabelsMasker(labels_img=atlas_filename, smoothing_fwhm=preproc_parameters['smoothing_fwhm'], 
                standardize=preproc_parameters['standardize'],
                detrend=preproc_parameters['detrend'], 
                low_pass=preproc_parameters['low_pass'],
                high_pass=preproc_parameters['high_pass'],
                t_r=preproc_parameters['t_r'],verbose=5, memory_level=1)

        if plotting_atlas is True:
            msgtitle = "Selected atlas:{}".format(atlas_name)
            plot_roi(atlas_filename, title=msgtitle)
        elif plotting_atlas_probabilistic is True:
            dmn_nodes = image.index_img(atlas_filename, [3, 4, 5, 6])
            plot_prob_atlas(dmn_nodes, cut_coords=(0, -55, 29), title="MSDL probabilistic atlas DMN")
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
    elif mask_type == 'brain-wide':        
        #create masker for seed analysis 
        #template = load_mni152_template()
        if epi_filename is None:
            print "Generating mask for brain-wide using datasets.fetch_icbm152_2009()" 
            icbms = datasets.fetch_icbm152_2009()
            masker = NiftiMasker(mask_img=icbms.mask, detrend=preproc_parameters['detrend'],
                                 standardize=preproc_parameters['standardize'],
                                 smoothing_fwhm=preproc_parameters['smoothing_fwhm'],
                                 low_pass=preproc_parameters['low_pass'],
                                 high_pass=preproc_parameters['high_pass'],
                                 t_r=preproc_parameters['t_r'],verbose=5)
        else:
            print "Extracting mask for raw EPI data:%s" %(epi_filename)
            mean_img = image.mean_img(epi_filename)
            template = load_mni152_template()
            #smoothed_img = image.smooth_img(epi_filename) 
            icbms = datasets.fetch_icbm152_2009()
            reshape_mask = resample_to_img(icbms.mask, template, interpolation='nearest')
            masker = NiftiMasker(mask_img= reshape_mask, mask_strategy = 'epi', detrend=preproc_parameters['detrend'],
                                 standardize=preproc_parameters['standardize'],
                                 smoothing_fwhm=preproc_parameters['smoothing_fwhm'],
                                 low_pass=preproc_parameters['low_pass'],
                                 high_pass=preproc_parameters['high_pass'],
                                 t_r=preproc_parameters['t_r'],verbose=5)
            
            
            masker.fit(epi_filename)
            plot_roi(masker.mask_img_, mean_img, title='EPI automatic mask')
                                                                
    return masker  

def motion_correction(epi_file, preproc_params):
    #from nipype.interfaces import fsl
    import subprocess 
    #import ntpath
    
    #mcflt = fsl.MCFLIRT()
    #mcflt.inputs.in_file = epi_file
    #mcflt.inputs.cost = 'mutualinfo'
    #mcflt.inputs.terminal_output = 'stream'
    #mcflt.inputs.stats_imgs = True
    
    dirname = os.path.dirname(epi_file)
    basename = os.path.basename(epi_file)
    mcfoutput = os.path.basename(epi_file)[:5] + 'outliers.txt'
    mcf_results_path = os.path.join(dirname,'mcf_results')
    if not os.path.exists(mcf_results_path):
        print('Creating mcf results path at {}',mcf_results_path)
        os.makedirs(mcf_results_path)

    mcfoutput = os.path.join('mcf_results', str(mcfoutput))
    mcfoutliers = os.path.basename(epi_file)[:5] + 'report.txt'
    mcfoutliers = os.path.join('mcf_results', str(mcfoutliers))
    # epi_file_output must exist 'touch epi_file_output'
    epi_file_output = os.path.join(dirname, str(epi_file))
    #mcflt.inputs.out_file = epi_file
    #mcflt.inputs.out_file = epi_file_output
    #mcflt.inputs.output_type = 'NIFTI'
    print "Computing Motion correction: mcflirt -in {} -cost mutualinfo -report -verbose \n".format(epi_file_output)
    # -out, -o <outputfile> default output file is infile_mcf
    #res = mcflt.run()
    # If the return code was non-zero it raises a CalledProcessError. 
    # The CalledProcessError object will have the return code in the returncode 
    # attribute and any output in the output attribute.
    
    # FSL Motion Outliers https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSLMotionOutliers
    # subprocess.check_output(["fsl_motion_outliers", "-i", str(epi_file_output), "-o", "output_motion_outliers", "-v"])
    mcf_outputfilename = os.path.basename(epi_file_output)[:-4] + '_mcf.' + 'nii.gz'
    subprocess.check_output(["mcflirt", "-in", str(epi_file_output), "-o", mcf_outputfilename,"-cost", "mutualinfo", "-report"])

    print("Calculating the outlier run and saving the report...")
    print "fsl_motion_outliers -i {} --dummy=4 --nomoco -o {} -v > {}".format(mcf_outputfilename, mcfoutput, mcfoutliers)
    #subprocess.Popen(["fsl_motion_outliers", "-i", mcf_outputfilename, "--nomoco", "-v >", mcfoutliers])
    out = subprocess.check_output(["fsl_motion_outliers", "-i", str(mcf_outputfilename), "--dummy=4", "--nomoco", "-o", str(mcfoutput), "-v >", str(mcfoutliers)])
    f = open( mcfoutliers, 'w' )
    f.write( out + '\n' )
    f.close()
    return True

def slicetime_correction(epi_file, preproc_params):
    '''slicetime_correction performs slice timing correction interleaved calling ot FSL
    With an interleaved slice order It's very controversial about if slice timing 
    should be used, especially when there is a severe head motion.'''
    import subprocess
    import ntpath
    imgpath = ntpath.split(epi_file)
    dirname = imgpath[0]
    filename = imgpath[1]
    extension = filename.split(".",1)[1]
    extension = "." + extension 
    filename = filename.split(".",1)[0]
    epi_file_output = filename +'_stc'
    epi_file_output = epi_file_output + extension
    
    epi_file_output = os.path.join(dirname, str(epi_file_output))
    do_stc = True
    if os.path.exists(epi_file_output):
        print "WARNING: A slice time correction version of file {} exists\n"
        yes = set(['yes','y', 'ye', ''])
        #no = set(['no','n'])
        print "Do you want to overwrite it? Y|N?"
        do_stc = raw_input('Do you want to overwrite it? Y|N?').lower()
        if do_stc in yes:
            do_stc =  True
        else:
            do_stc = False
            print "Exiting function, do not prforming Slice time Corection \n" 
            
    if do_stc is True:
        print "Computing Slice Timing correction: slicetimer -i {} [-o <corrected_timeseries>] [options] \n".format(epi_file_output)
        subprocess.check_output(["slicetimer", "-i", str(epi_file), "-o", str(epi_file_output), "-r", "2.5", "--odd", "-v"])
        return True
    else:
        return False
    
def extract_timeseries_from_mask(masker, epi_file):
    ''' extract time series from mask object for image defined in epi_file
    Input: masker is a mask built in the function generate_mask
    epi_file: is a image file
    if it is just one file it returns one ndarray time x voxels'''      
    print "'........Extracting time series for image {}:".format(epi_file)
    time_series = masker.fit_transform(epi_file)        
    print('Number of time points:', time_series.shape[0], 'Number of voxels:', time_series.shape[1])      
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
        print "Time series for only one subject"
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

def build_sparse_invariance_matrix(time_series, label_map=None, cohort=None):
    '''calculates the sparse covariance and precision matrices matrix for a GROUP of subjects
    Input: time_series ndarray subjects x time points x voxels and plots the connectome
    label_map: dict keys and values'''
    from nilearn import plotting
    from nilearn.connectome import GroupSparseCovarianceCV
    
    print('Calling to nilearn.connectome.GroupSparseCovarianceCV \
        Sparse inverse covariance w/ cross-validated choice of the parameter')
    gsc = GroupSparseCovarianceCV(verbose=2)
    gsc.fit(time_series)
    precision_matrix = -gsc.precisions_[...,0]
    covariances_matrix = gsc.covariances_[...,0]   
    return precision_matrix, covariances_matrix

def build_seed_based_coherency(seed_ts, nonseed_ts, preproc_parameters):
    ''' build_seed_based_coherency analysis
    Input: seed_ts ndarray of time series extracted from mask
    Input: nonseed_ts ndarray of time series extracted from entire brain mask
    Input: index_seed int 0 for PCC in DMN'''
    import nitime.timeseries as ts
    import nitime.analysis as nta
    methods = (None,
               {"this_method": 'welch', "NFFT": preproc_parameters['NFFT']},
               {"this_method": 'multi_taper_csd'},
               {"this_method": 'periodogram_csd', "NFFT": preproc_parameters})
    T_seed1 = ts.TimeSeries(seed_ts.T, sampling_interval=preproc_parameters['t_r']) #preproc_parameters['low_pass'] 
    T_target = ts.TimeSeries(nonseed_ts.T, sampling_interval=preproc_parameters['t_r'])
    T_target = ts.TimeSeries(np.vstack([seed_ts.T,nonseed_ts.T ]), sampling_interval=preproc_parameters['t_r'])
    this_method = methods[1] #welch method
    print "Computing Coherency ...."
    coh_seed = nta.SeedCoherenceAnalyzer(T_seed1, T_target,method=this_method, ub=0.1)
    return coh_seed
       
def build_seed_based_correlation(seed_ts, nonseed_ts, preproc_parameters):
    ''' build_seed_based_correlation from seed time series and nonseed time series
    Input: seed_ts, nonseed_ts type ndarray
    Output: reruns the fisher correlation of the nonseed time series and the one voxel (seed)'''
    print "Seed time series: time points=%s. nb of voxels=%s " % (len(seed_ts), seed_ts[0].shape)
    print "Non-Seed time series:  time points=%s. nb of voxels=%s" % (len(nonseed_ts), nonseed_ts[0].shape)

    seed_based_correlations = np.dot(nonseed_ts.T, seed_ts) / \
                          seed_ts.shape[0]
    print "seed-based correlation shape", seed_based_correlations.shape
    print "seed-based correlation normalized [-1,1]: min =", seed_based_correlations.min(), " max = ", seed_based_correlations.max()
    #Fisher-z transform the data to achieve a normal distribution. 
    #The transformed array can now have values more extreme than +/- 1.
    seed_based_correlations_fisher_z = np.arctanh(seed_based_correlations)
    print "seed-based correlation Fisher-z transformed: min =", seed_based_correlations_fisher_z.min(), \
    " max =", seed_based_correlations_fisher_z.max()                                                                                             
    return seed_based_correlations_fisher_z,seed_based_correlations

def plot_seed_based_correlation_MNI_space(seed_co, nonseed_masker, seed_coords, dirname, threshold, subject_id=None, cohort=None):
    '''plot_seed_based_correlation plots the seed based correlation in a brain in MNI space, it does an inverse transform'''
    from nilearn import plotting
    from nilearn import datasets
    from nilearn.input_data import NiftiMasker 

    #filename ='seedcorrelation.nii.gz'
    filename = "seedcorrelation_subject_{}.nii.gz".format(subject_id)
    msgtitle_prefix = 'Correlation'
    seed_based_correlation_img = nonseed_masker.inverse_transform(seed_co.T)
        
    imageresult = os.path.join(dirname, filename)
    seed_based_correlation_img.to_filename(imageresult)
    
    #pcc_coords = [(0, -52, 18)]
    #MNI152Template = '/usr/local/fsl/data/standard/MNI152_T1_2mm_brain_symmetric.nii.gz'
    #remove out-of brain functional connectivity using a mask
    icbms = datasets.fetch_icbm152_2009()
    masker_mni = NiftiMasker(mask_img=icbms.mask)
    data = masker_mni.fit_transform(imageresult)
    masked_sbc_z_img = masker_mni.inverse_transform(data)
    msgtitle = "Seed_{}_{}_G:{}_S:{}_thr:{}".format(msgtitle_prefix, seed_coords, cohort, subject_id, threshold)
    msgoutputfile = 'corr_map_' + 't:'+ str(threshold)[:4] + 's:'+ str(subject_id) + '.' + 'png'
    print('Saving correlation map at:{}',msgoutputfile)

    msgoutputfile = 'figures/' + msgoutputfile
    display = plotting.plot_stat_map(masked_sbc_z_img , cut_coords=seed_coords, output_file= msgoutputfile, \
                                         threshold=threshold, title= msgtitle, dim='auto', display_mode='ortho')
                                         
def plot_seed_based_coherence_MNI_space(Cxymean, nonseed_masker, seed_coords, dirname, threshold, subject_id=None, cohort=None):
    ''' plot_seed_based_coherence in MNI space using an inverse transform
    Input: Cxymean the coherence between all voxel pairs, type 'numpy.ndarray'
    nonseed_masker: Mask of the non seed, the entire brain
    seed_coords: coordinates of the seed,tuple, typically the PCC (0, -52, 18),
    dirname path where the image is saved
    threshold: threshold to slect relevant coherence
    subject_id and cohort strings for the masgtitle of the plot'''
    from nilearn import plotting
    from nilearn import datasets
    from nilearn.input_data import NiftiMasker 
    filename = "seedcoherence_subjec_{}.nii.gz".format(subject_id)
    msgtitle_prefix = 'Coherence'
    seed_based_correlation_img = nonseed_masker.inverse_transform(Cxymean.T)
    imageresult = os.path.join(dirname, filename)
    seed_based_correlation_img.to_filename(imageresult)
    #pcc_coords = [(0, -52, 18)]
    #MNI152Template = '/usr/local/fsl/data/standard/MNI152_T1_2mm_brain_symmetric.nii.gz'
    #remove out-of brain functional connectivity using a mask
    icbms = datasets.fetch_icbm152_2009()
    masker_mni = NiftiMasker(mask_img=icbms.mask)
    data = masker_mni.fit_transform(imageresult)
    masked_sbc_z_img = masker_mni.inverse_transform(data)
    msgtitle = "Seed_{}_{}_G:{}_S:{}_thr:{}".format(msgtitle_prefix, seed_coords, cohort, subject_id, threshold)
    msgoutputfile = 'coh_map_' + 't:'+str(threshold)[:4] + 's_'+ str(subject_id) + '.png'
    msgoutputfile = 'figures/' + msgoutputfile
    display = plotting.plot_stat_map(masked_sbc_z_img , cut_coords=seed_coords, output_file= msgoutputfile, \
                                         threshold=threshold, title= msgtitle, dim='auto', display_mode='ortho')
    return display
    
def build_connectome(time_series, kind_of_correlation):
    ''' build_connectome: computes different kinds of functional connectivity matrices 
    for the time series according to kind_of_analysis and
    kind_of_correlation:{“correlation”, “partial correlation”, “tangent”, “covariance”, “precision”} '''
    from nilearn.connectome import ConnectivityMeasure 
    from sklearn.covariance import LedoitWolf, EmpiricalCovariance
    # LedoitWolf estimatorhas slightly shrunk towards zero compared to a maximum-likelihood estimate
    typeofestimator = ['empirical', 'Ledoit']
    idx = 1
    if typeofestimator[idx] is 'empirical':
                      connectivity_measure = ConnectivityMeasure(EmpiricalCovariance(assume_centered=True), kind=kind_of_correlation) 
    if typeofestimator[idx] is 'Ledoit':
                      connectivity_measure = ConnectivityMeasure(LedoitWolf(assume_centered=True), kind=kind_of_correlation) 
    correlation_matrix_subjects = []
    for i in range(0, len(time_series)):
        correlation_matrix = connectivity_measure.fit_transform([time_series[i]])[0]
        correlation_matrix_subjects.append(correlation_matrix)
    print "Built correlation matrices:{}, Estimator:{}, Dimension {}x{}x{}\n".format(kind_of_correlation, typeofestimator[idx], len(correlation_matrix_subjects), correlation_matrix_subjects[0].shape[0], correlation_matrix_subjects[0].shape[1])    
    arr_corr_matrices = np.array(correlation_matrix_subjects)
    print "The wise element mean of the correlation matrices ={}\n".format(arr_corr_matrices.mean(axis=0)) 
    means = [np.mean([el for el in sublist]) for sublist in correlation_matrix_subjects]
    arrmeans = np.array(means)
    print "The overall mean of the correlation matrix for each subject is={}\n".format(arrmeans)
    print "The mean across subjects is %.3f and the std is %.3f\n", (np.mean(arrmeans), np.std(arrmeans))    
    return correlation_matrix_subjects

def build_connectome_in_frequency(time_series, preproc_params, freqband=None):
    ''' build_connectome_in_frequency computes functional connectivity matrices
    based on coherency
    Input: time_series : subjects x time points x voxels
         : preproc_params
         : freqband: [high pass, low pass] eg 0.01, 0.1''' 
    time_series.ndim
    subjects_matrices = []
    for s in range(0, time_series.shape[0]):
        print "Calculating connectome coherency based matrix for subject: {} /  {}".format(s, time_series.shape[0]-1)
        # initialize voxels x voxels
        coherency_1s = np.zeros((time_series.shape[2], time_series.shape[2]))
        #coherency_1s = np.fill_diagonal(coherency_1s, 1)
        for voxi in range(0,time_series.shape[2]):
            for voxj in range(0,time_series.shape[2]):
                #if voxj >= voxi: or make the matrix symmetric at the end
                x = time_series[s].T[voxi]
                y = time_series[s].T[voxj]
                f, cxy = calculate_coherence(x,y, preproc_params)
                if freqband is None:
                    coherency_1s[voxi, voxj] = np.mean(cxy)
                else:
                    maskfreqs = (f>=freqband[0]) & (f <=freqband[1])
                    coherency_1s[voxi, voxj] = np.mean(cxy[maskfreqs])
                    print "Coherence at {}x{}= {}".format(voxi, voxj,coherency_1s[voxi, voxj] )   
        subjects_matrices.append(coherency_1s)      
    
    return subjects_matrices            
                
    
def plot_correlation_matrix(corr_matrix, label_map, coord_map, msgtitle, what_to_plot=None, edge_threshold=None):
    ''' plot correlation matrix
    Input: ONE correlation matrix
    label_map : dict with rois (label_map.keys()) and values (label_map.values())
    masgtitle'''
    from nilearn import plotting
    from nitime.viz import drawmatrix_channels, drawgraph_channels
    # if what_to_plot is not specified define what to plot
    if what_to_plot is None:
        plot_heatmap = True
        plot_graph = True
        plot_connectome = True
    else:
        plot_heatmap = what_to_plot['plot_heatmap']
        plot_graph = what_to_plot['plot_graph']
        plot_connectome = what_to_plot['plot_connectome']       

    if plot_heatmap is True:
        print('Plotting correlation_matrix as a heatmap from nitime...')
        #fig_h_drawx = drawmatrix_channels(corr_matrix, label_map[0], size=[10., 10.], color_anchor=0, title= msgtitle)  
        fig_h_drawx = drawmatrix_channels(corr_matrix, label_map, size=[10., 10.], color_anchor=0, title= msgtitle)   
    if plot_graph == True:    
        print('Plotting correlation_matrix as a (circular) network nitime...')
        #fig_g_drawg = drawgraph_channels(corr_matrix, label_map[0],title=msgtitle)
        fig_g_drawg = drawgraph_channels(corr_matrix, label_map,title=msgtitle)
    #plotting connectivity network with brain overimposed
    if plot_connectome is True:
        if edge_threshold is None:
            edge_threshold = 0 # plot all edges with intensity automatic
        #fig_c_drawg = plotting.plot_connectome(corr_matrix, label_map[1],edge_threshold=edge_threshold, title=msgtitle,display_mode="ortho") #,edge_vmax=.5, edge_vmin=-.5       
        fig_c_drawg = plotting.plot_connectome(corr_matrix, coord_map,edge_threshold=edge_threshold, title=msgtitle,display_mode="ortho") 
        
            #if  what_to_plot['plot_heatmap'] is True:
    #if  what_to_plot['plot_graph'] is True:
    #if  what_to_plot['plot_connectome'] is True:    
    #    plotting.plot_connectome(precision_matrix, label_map.values(), edge_threshold=edge_threshold,
    #                         title=str(edge_threshold)+'-GroupSparseCovariancePrec', display_mode='lzr')
    #    plotting.plot_connectome(covariances_matrix, label_map.values(), edge_threshold=edge_threshold,
    #                         title=str(edge_threshold)+'-GroupSparseCovariance', display_mode='lzr')
    #    msgtitle = "Precision matrix:%s, edge threshold=%s" % (cohort,edge_threshold) 
        #plot only the heat map
    #    plot_correlation_matrix(precision_matrix,label_map,msgtitle)
        #plot_covariance_matrix(gsc.covariances_[..., 0],gsc.precisions_[..., 0], labels, title = str(edge_threshold)+"-GroupSparseCovariance")
    #    plotting.show()
    # persistent homology analysis
    #persistent_homology(gsc.covariances_[..., 0], coords)
    #pdb.set_trace()
def calculate_coherence(x,y, preproc_params):
    ''' Estimate the magnitude squared coherence estimate, Cxy, of discrete-time signals X and Y using Welch’s method.
    Input: signals x and y ndarray samples x time points. Calculates the coherence of x for each signal in y 
    x: (1 sample, time points) and y: (n samples, time points) in this shape use axis by default axis = -1'''
    # nperseg Length of each segment, if large returns 1, by default is 256 more than out segment will return 1
    # minimum window length can be pick the time lag where the autocorrelation *first* drops 
    nperseg= 16
    noverlap = 8
    # Axis along which the coherence is computed for both inputs; the default is over the last axis (i.e. axis=-1).
    f, Cxy = signal.coherence(x, y, fs=preproc_params['fs'], nfft=preproc_params['nfft'], nperseg=nperseg, noverlap=noverlap) 
    #calculate spectral connectivity measures using MNE
    #from mne.connectivity import spectral_connectivity
    #spectral_connectivity()
    return f, Cxy


def fourier_spectral_estimation(ts, image_params, msgtitle=None):
    """Calculate the PSD estimate from a time series using Welch's method
    The power spectrum calculates the area under the signal plot using the discrete Fourier Transform
    The PSD assigns units of power to each unit of frequency and thus, enhances periodicities 
    Welch’s method computes an estimate of the power spectral density by dividing 
    the data into overlapping segments, computing a modified periodogram for each segment and averaging the periodograms.
    by default constant detrend, one-sided spectrum, 
    noverlap = nperseg / 2 (noverlap is 0, this method is equivalent to Bartlett’s method). 
    scaling 'density'V**2/Hz 'spectrum' V**2.
    returns array of sampling frerquencies and the power spectral density of the ts
    Input: ts is a ndarray of time series (samples x time points)
    Example: f, Pxx_den = fourier_spectral_estimation(ts)    
    """
     
    #data_img, image_params = load_fmri_image_name()
    #data = np.zeros(1, 1)  # initialize for number of rois and number of samples
    # The cadillac of spectral estimates is the multi-taper estimation, 
    # which provides both robustness and granularity, but notice that 
    # this estimate requires more computation than other estimates  
    psd_results = []
    plotPxx = True
    if plotPxx is True: 
        fig, ax = plt.subplots(ts.shape[0], 1, sharex=True, sharey=True, squeeze=False)    
    for i in range(0, ts.shape[0]):
        mnicoords = get_MNI_coordinates('DMN')
        msgtitlepre = "{}".format('DMN')
        #nperseg is the length of each segment, 256 by default
        nperseg = 16
        f, Pxx_den = signal.welch(ts[i,:], image_params['fs'], nperseg=nperseg, detrend='constant', nfft =image_params['nfft'], scaling = 'density')  
        pxx = [f, Pxx_den]
        #idx = (f>=image_params['high_pass'])*(f<=image_params['low_pass'])
        #f = f[idx]
        #Pxx_den = Pxx_den[idx]
        #pxx = [f, Pxx_den]
        psd_results.append(pxx)
        print "Timeseries:", i," frequency sampling", f, " Pxx_den:", Pxx_den, " Max Amplitude is:", np.mean(Pxx_den.max())
        if plotPxx is True:
            #plt.figure()
            #ax[i].semilogy(f, Pxx_den)
            ax[i,0].plot(f, Pxx_den)
            if i == ts.shape[0]-1:
                ax[i,0].set_xlabel('frequency [Hz]')
            ax[i,0].set_ylabel('PSD [V**2/Hz]')
            msgtitlepost = "{}_node:{}".format(msgtitlepre, mnicoords.keys()[i])
            ax[i,0].set_title(msgtitlepost)  
    print("Saving PSD results")    
    save_plot(ax, 'subject:' + msgtitle)
    return  psd_results 

def save_plot(ax, msgtitle):
    figsdirectory = '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_control/figures/' 
    figsdirectory = os.getcwd()
    filenametosave = os.path.join('figures/','psd_' + msgtitle)

    print("Saving PSD at {} \n",filenametosave)
    if os.path.exists(figsdirectory) is True:
        plt.savefig(filenametosave, bbox_inches="tight")
    else:
        try: 
            os.makedirs(figsdirectory)
            plt.savefig(filenametosave, bbox_inches="tight")
        except OSError:
            if not os.path.isdir(figsdirectory):
                raise 
    plt.close()

def plot_nii(nifti=None):
    """ plot nifti image from the path
    """
    from nilearn import plotting
    if nifti is None:
        nifti = 'seedcorrelation_subject_Mean.nii.gz'
    plotting.plot_stat_map(tmap_filename)
    # to plot 4D image for example ICa image
    ica_nii = '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_scdplus/CanICA_resting_state.nii.gz'
    plotting.plot_prob_atlas(ica_nii)