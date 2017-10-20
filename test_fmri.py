#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 11:25:08 2017

@author: jaime
The test_fmri.py is a module that tests the Module analysis_fmri.py 
containing preprocessing and anaylis of fMRI images
"""
import os
import numpy as np
import pdb
import matplotlib.pyplot as plt
import analysis_fmri as afmri
print('Calling to test_fmri to test functions in analysis_fmri')
#Load a list of images load_epi_images(epi_file_list=None, subjects_list=None, dir_name=None, f_name=None)
#The location of the nifti iamges can be input
#literally epi_file_list or suing the base directory the name of the subjects and the fname
 

        
def select_cohort(cohort_group):
    '''load a list of nifti images defined in cohort_group'''
    epi_file_list_conv = ['/Users/jaime/vallecas/data/converters_y1/converters/w0015_fMRI.nii',
                          '/Users/jaime/vallecas/data/converters_y1/converters/w0242_fMRI.nii',
                          '/Users/jaime/vallecas/data/converters_y1/converters/w0277_fMRI.nii',
                          '/Users/jaime/vallecas/data/converters_y1/converters/w0377_fMRI.nii',
                          '/Users/jaime/vallecas/data/converters_y1/converters/w0464_fMRI.nii',
                          '/Users/jaime/vallecas/data/converters_y1/converters/w0637_fMRI.nii',
                          '/Users/jaime/vallecas/data/converters_y1/converters/w0885_fMRI.nii',
                          '/Users/jaime/vallecas/data/converters_y1/converters/w0090_fMRI.nii',
                          '/Users/jaime/vallecas/data/converters_y1/converters/w0243_fMRI.nii',
                          '/Users/jaime/vallecas/data/converters_y1/converters/w0290_fMRI.nii',
                          '/Users/jaime/vallecas/data/converters_y1/converters/w0382_fMRI.nii',
                          '/Users/jaime/vallecas/data/converters_y1/converters/w0583_fMRI.nii',
                          '/Users/jaime/vallecas/data/converters_y1/converters/w0707_fMRI.nii',
                          '/Users/jaime/vallecas/data/converters_y1/converters/w0938_fMRI.nii',
                          '/Users/jaime/vallecas/data/converters_y1/converters/w0940_fMRI.nii',
                          '/Users/jaime/vallecas/data/converters_y1/converters/w0208_fMRI.nii',
                          '/Users/jaime/vallecas/data/converters_y1/converters/w0263_fMRI.nii',
                          '/Users/jaime/vallecas/data/converters_y1/converters/w0372_fMRI.nii',
                          '/Users/jaime/vallecas/data/converters_y1/converters/w0407_fMRI.nii',
                          '/Users/jaime/vallecas/data/converters_y1/converters/w0618_fMRI.nii',
                          '/Users/jaime/vallecas/data/converters_y1/converters/w0882_fMRI.nii',
                          '/Users/jaime/vallecas/data/converters_y1/converters/w1021_fMRI.nii']                
    epi_file_list_control = ['/Users/jaime/vallecas/data/converters_y1/controls/w0022_fMRI.nii',
                             '/Users/jaime/vallecas/data/converters_y1/controls/w0077_fMRI.nii',
                             '/Users/jaime/vallecas/data/converters_y1/controls/w0124_fMRI.nii',
                             '/Users/jaime/vallecas/data/converters_y1/controls/w0189_fMRI.nii',
                             '/Users/jaime/vallecas/data/converters_y1/controls/w0365_fMRI.nii',
                             '/Users/jaime/vallecas/data/converters_y1/controls/w0523_fMRI.nii',
                             '/Users/jaime/vallecas/data/converters_y1/controls/w0832_fMRI.nii',
                             '/Users/jaime/vallecas/data/converters_y1/controls/w1017_fMRI.nii',
                             #'/Users/jaime/vallecas/data/converters_y1/controls/w0243_fMRI.nii',
                             '/Users/jaime/vallecas/data/converters_y1/controls/w0028_fMRI.nii',
                             '/Users/jaime/vallecas/data/converters_y1/controls/w0105_fMRI.nii',
                             '/Users/jaime/vallecas/data/converters_y1/controls/w0171_fMRI.nii',
                             '/Users/jaime/vallecas/data/converters_y1/controls/w0312_fMRI.nii',
                             '/Users/jaime/vallecas/data/converters_y1/controls/w0429_fMRI.nii',
                             '/Users/jaime/vallecas/data/converters_y1/controls/w0691_fMRI.nii',
                             '/Users/jaime/vallecas/data/converters_y1/controls/w0841_fMRI.nii',
                             '/Users/jaime/vallecas/data/converters_y1/controls/w1116_fMRI.nii',
                             '/Users/jaime/vallecas/data/converters_y1/controls/w0049_fMRI.nii',
                             '/Users/jaime/vallecas/data/converters_y1/controls/w0121_fMRI.nii',
                             '/Users/jaime/vallecas/data/converters_y1/controls/w0187_fMRI.nii',
                             '/Users/jaime/vallecas/data/converters_y1/controls/w0337_fMRI.nii',
                             '/Users/jaime/vallecas/data/converters_y1/controls/w0450_fMRI.nii',
                             '/Users/jaime/vallecas/data/converters_y1/controls/w0805_fMRI.nii',
                             '/Users/jaime/vallecas/data/converters_y1/controls/w0935_fMRI.nii']  
    epi_file_list_one = ['/Users/jaime/vallecas/data/converters_y1/controls/w0022_fMRI.nii']
    if cohort_group is 'converter':               
        file_list = epi_file_list_conv 
    elif cohort_group is 'control':
        file_list = epi_file_list_control
    elif cohort_group is 'single_subject':
        #load a single image
        file_list = epi_file_list_one
    return file_list

def preproc_parameters_list():
    ''' Load preprocessign paarmeters as a dictionary  
    preproc_list.keys()
    preproc_list.values()'''
    preproc_list = {'standardize': 'True', 'detrend': 'True', 'smoothing_fwhm': 8, 't_r':2.5,
                    'low_pass': 0.1, 'high_pass': 0.01, 'verbose':2}
    print('The preprocessing parameters are:')
    print "preproc_list['standardize']: ", preproc_list['standardize']
    print "preproc_list['detrend']: ", preproc_list['detrend']
    print "preproc_list['t_r']: ", preproc_list['t_r']
    print "preproc_list['smoothing_fwhm']: ", preproc_list['smoothing_fwhm']
    print "preproc_list['low_pass']: ", preproc_list['low_pass']
    print "preproc_list['high_pass']: ", preproc_list['high_pass']
    return preproc_list

def verify_images_exist(file_list):
    '''check that the list of files deined in epi_file_list exist return True or False'''
    for f in file_list:
        if os.path.exists([file_list]) is not True:
            print('ERROR: The file', file_list[f], ' does not exist!!')
            return False
    print('epi_file_list found')
    
    return True   

def verify_and_load_images(file_list=None, subjects_list=None, dir_name=None, f_name=None):
    '''load a list of epi images which can be given as a list of nifit images
    or built from subjects list and dirname parameters
    Example: load_epi_images([niftipath1,niftipath2....]) returns the same path
    Example: oad_epi_images([subject1, subject2], directoryofsubjects) returns a list 
    with the nifti images in the directory directoryofsubjects/subject*
    '''
    print('Verifying that the images exist...')
    epi_file_list = []
    if file_list is not None:
        epi_file_list = file_list
        for f in file_list:
            if os.path.isfile(f) is False:
                print('ERROR loading the image:', f, ' the file doesnt exist!!')
                epi_file_list = []
                break
    else:
        #build path the list of nifti images from each subject in dirname/[subjects_list]
        if (int(subjects_list is None) + int(dir_name is None)+ int(f_name is None) <1):
            for i in range(0,len(subjects_list)):
                subjname = os.path.join(dir_name,subjects_list[i])
                epi_file_list.append(os.path.join(subjname, f_name)) 
        else:
            print('ERROR loading the images')
    return epi_file_list

def granger_causality_analysis(time_series, preproc_parameters_list,label_map,order):
    ''' granger test and plot the connectivity matrix'''
    #testing the null hypothesis of no granger
    print("Testing for Granger causality")
    granger_test_results = afmri.test_for_granger(time_series, preproc_parameters_list, label_map, order)
    #building granger connectome
    afmri.build_granger_matrix(time_series, preproc_parameters_list, label_map)
    return granger_test_results




     
group = ['converter', 'control', 'single_subject']
cohort = group[1]
file_list = select_cohort(cohort)

#verify and load can be load from the full path of each images verify_and_load_images(load_list_of_epi_images(cohort))
#or using the subjects id for a hierarchical std structure: verify_and_load_images(None, '['bcpa0537_1','bcpa0578_1', 'bcpa0650_1']','/Users/jaime/vallecas/mario_fa/RF_off','wbold_data.nii')
epi_file_list = verify_and_load_images(file_list)
if len(epi_file_list) > 0:
    print('Nifti images for cohort:', cohort, ' loaded')
    print(epi_file_list)
     
else:
    print('ERROR: File(s) containing the images exist')
# Load preprocessign parameters   
preproc_parameters_list = preproc_parameters_list()

mask_type = ['atlas','DMN', 'AN', 'SN']
idx = 0

if mask_type[idx] == 'atlas':
    mask_label = 'cort-maxprob-thr25-2mm'
    #mask_label = 'power_2011'
    label_map = afmri.get_atlas_labels(mask_label)
elif mask_type[idx] == 'DMN':
    mask_label= afmri.get_MNI_coordinates(mask_type[idx])   
    label_map = [afmri.get_MNI_coordinates(mask_type[idx]).keys(), afmri.get_MNI_coordinates(mask_type[idx]).values()]   
    #coords_map = afmri.get_MNI_coordinates(mask_type[idx]).values()  
  
masker = afmri.generate_mask(mask_type[idx], mask_label, preproc_parameters_list)
print "The masker parameters are:%s\n" % (masker.get_params())
#extract time series for the masker with the preproc parameters
time_series = afmri.extract_timeseries_from_mask(masker, epi_file_list)

nb_of_subjects = len(time_series)
print "Time series dimensions. subjects=%d" %nb_of_subjects
pdb.set_trace()
# Precision matrix
print "Calculating the Covariance and the Precision Matrix (inverse covariance)"
precision_matrix = afmri.build_sparse_invariance_matrix(time_series, label_map)


# Granger causality : test and plot the granger connectome 
print "Calculating granger causality matrix, subjects:%d Mask type:%s" %(nb_of_subjects, mask_type[idx])
#granger_test_results = granger_causality_analysis(time_series, preproc_parameters_list,label_map, order=10)

# correlation analysis
kind_of_correlation = ['correlation', 'covariance', 'tangent', 'precision', 'partial correlation']
idcor = 0
corr_matrices = afmri.build_correlation_matrix(time_series, kind_of_analysis='time', kind_of_correlation=kind_of_correlation[idcor])

#pass one correlation matrix to plot
subject_id = None
msgtitle = "Subject_{}, Group:{}, Mask:{}, Corr:{}".format(subject_id, cohort, mask_label,kind_of_correlation[idcor] )
if type(corr_matrices) is list:
    corr_matrices = np.transpose(np.asarray(corr_matrices))
    corr_matrices_mean = corr_matrices.mean(-1)
# plot mean of correlation matrices, to plot single  individual corr_matrices[subjectid]
# to plot mrean use corr_matrices_mean        
pdb.set_trace()
afmri.plot_correlation_matrix(corr_matrices_mean,label_map, msgtitle)

# Seed_based correlation
print "Calculting seed based correlation"
build_seed_based_correlation_matrix(masker.fit_transform(masker, epi_file, time_series, preproc_parameters, nonseed_mask='brain-wide')






pdb.set_trace()
#Group ICA analysis
print('Calling to group ICA...')
#pdb.set_trace()
afmri.group_ICA(epi_file_list, preproc_parameters_list, cohort)

print('Calling to group Ward clustering...')
afmri.clustering_Ward(epi_file_list, preproc_parameters_list, cohort)