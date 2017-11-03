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
from collections import OrderedDict
from nilearn import plotting
from nilearn import datasets
from nilearn.input_data import NiftiMasker 
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


def plot_time_series(timeseries, msgtitle=None):
    ''' plot time series'''
    plt.plot(timeseries)
    if msgtitle is None:
        msgtitle = "time series SubjectId:{}".format(msgtitle)
    plt.title(msgtitle)
    plt.xlabel('Scan number')
    plt.ylabel('Normalized signal')
    
    
def preproc_parameters_list():
    ''' Load preprocessign paarmeters as a dictionary  
    preproc_list.keys()
    preproc_list.values()'''
    preproc_list = {'standardize': 'True', 'detrend': 'True', 'smoothing_fwhm': 8, 't_r':2.5,
                    'low_pass': 0.2, 'high_pass': 0.01, 'fs': 0.4, 'nfft':129 , 'verbose':2}
    print('The preprocessing parameters are:')
    print "preproc_list['standardize']: ", preproc_list['standardize']
    print "preproc_list['detrend']: ", preproc_list['detrend']
    print "preproc_list['t_r']: ", preproc_list['t_r']
    print "preproc_list['smoothing_fwhm']: ", preproc_list['smoothing_fwhm']
    print "preproc_list['low_pass']: ", preproc_list['low_pass']
    print "preproc_list['high_pass']: ", preproc_list['high_pass']
    print "preproc_list['fs']: ", preproc_list['fs']
    print "preproc_list['nfft']: ", preproc_list['nfft']
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

def extract_seed_ts(time_series,seed_id):
    ''' extract_seed_mask_and_ts
    Input: time series ndarray
    Input: seed_id 0 for PCC in DMN
    Output: seed_ts'''
    
    seed_ts = time_series[:,seed_id].reshape(time_series.shape[0],1) 
    return seed_ts

def extract_non_seed_mask_and_ts(epi_file, preproc_parameters_list):    
    ''' extract_non_seed_mask_and_ts '''
    nonseed_masker = afmri.generate_mask('brain-wide', [], preproc_parameters_list)
    nonseed_ts = afmri.extract_timeseries_from_mask(nonseed_masker, epi_file) 
    return nonseed_masker, nonseed_ts
    
def calculate_and_plot_seed_based_correlation(time_series, nonseed_masker,nonseed_ts, mask_type,mask_label,preproc_parameters_list,epi_file,seed_coords,seed_id, dirname, cohort):
    ''' calculate_seed_based_correlation and plot the contrast in MNI for one subject'''
    print "Calculting seed based correlation: one Seed vs. Entire Brain"
    # seed_ts dimension is timepints x nb of seeds (120x1)
    seed_ts = time_series[:,seed_id].reshape(time_series.shape[0],1) 
    # generate brain-wide masker from fMRI epi_file[subjectr_id]
    #nonseed_masker = afmri.generate_mask('brain-wide', [], preproc_parameters_list)
    #nonseed_ts = afmri.extract_timeseries_from_mask(nonseed_masker, epi_file_list) 
    #nonseed_ts = afmri.extract_timeseries_from_mask(nonseed_masker, epi_file) 
    # compute the seed based correlation between seed and nonseed time series per each subject
    [seed_corr_fisher,seed_corr]  = afmri.build_seed_based_correlation(seed_ts, nonseed_ts, preproc_parameters_list)
    threshold = 0.6
    # plot via inverse trnasform the correlation
    afmri.plot_seed_based_correlation(seed_corr, nonseed_masker, seed_coords, dirname, threshold,subject_id, cohort)
    return [seed_corr_fisher,nonseed_masker,nonseed_ts]
 
def calculate_and_plot_seed_based_coherence(time_series, nonseed_masker,nonseed_ts, mask_type, mask_label,preproc_parameters_list,epi_file,seed_coords,seed_id, dirname, cohort, typeofcorr=None):
    ''' calculate_and_plot_seed_based_coherence '''
    import pprint
    print "Calculting seed based coherence: one Seed vs. Entire Brain"
    #seed_ts = time_series[:,seed_id].reshape(time_series.shape[0],1) 
    # generate brain-wide masker from fMRI epi_file[subjectr_id]
    #nonseed_masker = afmri.generate_mask('brain-wide', [], preproc_parameters_list)
    # time series of entire brain
    #nonseed_ts = afmri.extract_timeseries_from_mask(nonseed_masker, epi_file) 
    # compute the seed based coherence between seed and nonseed time series per each subject
    
    print "Calculating the Seed based Coherence seed vs brain..."
    coh_seedbrain = afmri.build_seed_based_coherency(seed_ts, nonseed_ts, preproc_parameters_list)
    seed_coherence = coh_seedbrain.coherence
    print "Frequencies:"
    pprint.pprint(coh_seedbrain.frequencies)
    print "Coherence:"
    pprint.pprint(seed_coherence)
    print "Relative Phases"
    pprint.pprint(coh_seedbrain.relative_phases)
    # plot via inverse trnasform the coherence
    pdb.set_trace()
    afmri.plot_seed_based_correlation(coh_seedbrain, nonseed_ts, seed_coords, dirname, threshold,subject_id, cohort, typeofcorr)
    return coh_seedbrain

# Load preprocessign parameters   
preproc_parameters_list = preproc_parameters_list()
    
###########Testing fucntion space #########
seed_ts = np.load('seed_ts.npy')
print "Seed ts dimensions={} X {}".format(seed_ts.shape[0], seed_ts.shape[1])
nonseed_ts = np.load('nonseed_ts.npy')
print "NonSeed ts dimensions={} X {}".format(nonseed_ts.shape[0], nonseed_ts.shape[1]) 

#afmri.fourier_spectral_estimation(seed_ts.T, image_params=preproc_parameters_list(),msgtitle='converter')
#psd = afmri.fourier_spectral_estimation(nonseed_ts.T[10234:10239], image_params=preproc_parameters_list(),msgtitle='converter')
nb_voxels = nonseed_ts.shape[1]
nonseed_forcoh = nonseed_ts.T[12345:12349,:].reshape(4,120)
f, Cxy = afmri.calculate_coherence(seed_ts.T, nonseed_forcoh[0],preproc_parameters_list)
# Compute coherence for the entire nonseed_ts
#f, Cxy = afmri.calculate_coherence(seed_ts.T, nonseed_ts.T,preproc_parameters_list)
# Estimate the magnitude squared coherence estimate, Cxy, of discrete-time signals X and Y using Welch’s method.

            
pdb.set_trace()
###########Testing fucntion space #########
 
 
group = ['converter', 'control', 'single_subject']
cohort = group[0]
file_list = select_cohort(cohort)

#verify and load can be load from the full path of each images verify_and_load_images(load_list_of_epi_images(cohort))
#or using the subjects id for a hierarchical std structure: verify_and_load_images(None, '['bcpa0537_1','bcpa0578_1', 'bcpa0650_1']','/Users/jaime/vallecas/mario_fa/RF_off','wbold_data.nii')
epi_file_list = verify_and_load_images(file_list)
if len(epi_file_list) > 0:
    print('Nifti images for cohort:', cohort, ' loaded')
    print(epi_file_list)
    nb_of_subjects = len(epi_file_list)
    print "Nb of subjects =%s" % nb_of_subjects
    dirname = os.path.dirname(epi_file_list[0])
else:
    print('ERROR: File(s) containing the images exist')


mask_type = ['atlas','DMN', 'AN', 'SN', 'brain-wide']
idx = 1

if mask_type[idx] == 'atlas':
    mask_label = 'cort-maxprob-thr25-2mm'
    #mask_label = 'power_2011'
    label_map = afmri.get_atlas_labels(mask_label)
elif mask_type[idx] == 'DMN':
    mask_label= afmri.get_MNI_coordinates(mask_type[idx])   
    label_map = [mask_label.keys(), mask_label.values()]
    #label_map = [afmri.get_MNI_coordinates(mask_label.keys(), afmri.get_MNI_coordinates(mask_type[idx]).values()]   
    #coords_map = afmri.get_MNI_coordinates(mask_type[idx]).values()  
    
masker = afmri.generate_mask(mask_type[idx], mask_label, preproc_parameters_list)
print "The masker parameters are:%s\n" % (masker.get_params())
#extract time series for the masker with the preproc parameters
time_series = afmri.extract_timeseries_from_mask(masker, epi_file_list)
print ("The extracted time series is ", type(time_series))    
#plot time series

# Seed based analysis
seed_single_subject = True
seed_id = 0
seed_coords = label_map[1][seed_id]
threshold=0.6
if seed_single_subject is True:
    # Calculate seed based correlation per one subject
    subject_id = 4
    plot_time_series(time_series[subject_id], subject_id)
    # id of the seed. To get the PCC use the DMN label = ['PCC', 'lTPJ', 'rTPJ', 'mPFC']
    seed_ts = time_series[subject_id]

    seed_ts = extract_seed_ts(seed_ts,seed_id)
    print "Extracting time series from the entire brain....seat tight, it will take some time....\n"
    nonseed_masker, nonseed_ts = extract_non_seed_mask_and_ts(epi_file_list[subject_id], preproc_parameters_list)
    print "Calculatign the seed based correlation matrix and ploting in MNI space....\n"
    seed_corr_fisher = calculate_and_plot_seed_based_correlation(seed_ts, nonseed_masker,nonseed_ts,mask_type[idx], mask_label,preproc_parameters_list, epi_file_list[subject_id],seed_coords,seed_id,dirname, cohort)
    # seed based Coherency analysis
    print "Calculating the seed based coherency matrix and ploting in MNI space....\n"
    seed_coherence = calculate_and_plot_seed_based_coherence(seed_ts, nonseed_masker, nonseed_ts, mask_type[idx], mask_label,
                                                             preproc_parameters_list, epi_file_list[subject_id],seed_coords,seed_id,dirname, cohort, 'coherence')
    pdb.set_trace()
        
    
else:
    #loop 
    non_seed_corr_list = []
    non_seed_masker_list = []
    non_seed_ts_list = []
    #nb_of_subjects = 2
    for subject_id in range(0, nb_of_subjects):
        print "Extracting time series for Subject %s / %s" % (subject_id, nb_of_subjects-1)
        seed_ts = time_series[subject_id]
        plot_time_series(time_series[subject_id], subject_id)
        seed_ts = extract_seed_ts(seed_ts,seed_id)
        nonseed_masker, nonseed_ts = extract_non_seed_mask_and_ts(epi_file_list[subject_id], preproc_parameters_list)
        
        nonseed_corr_fisher, nonseed_masker,nonseed_ts = calculate_and_plot_seed_based_correlation(seed_ts, nonseed_masker, nonseed_ts,mask_type[idx],mask_label,preproc_parameters_list,
                                                            epi_file_list[subject_id],seed_coords,seed_id,dirname, cohort, 'coherence')
        non_seed_masker_list.append(nonseed_masker)
        non_seed_corr_list.append(nonseed_corr_fisher)
        non_seed_ts_list.append(nonseed_ts)
    #calculate the mean of the seed based across individuals
    # mean in absolute value
    #arr_fisher_corr = np.abs(np.array(non_seed_corr_list))
    arr_fisher_corr = np.array(non_seed_corr_list)
    print "Wise mean of the Fisher seed correlation across subjects. min=%s, max=%s, mean=%s and std=%s." % (arr_fisher_corr.min(), arr_fisher_corr.max(), arr_fisher_corr.mean(), arr_fisher_corr.std())
    wisemean_fisher = arr_fisher_corr.mean(axis=0)
    voxels = wisemean_fisher.shape[0]
    wisemean_fisher = wisemean_fisher.reshape(voxels)
    subject_id='Mean:'
    # save in file
    #np.save('/Users/jaime/vallecas/data/converters_y1/converters/results/conv_arr_fisher_corr', arr_fisher_corr)
    #conv_params_plot= [wisemean_fisher, non_seed_masker_list[0], seed_coords, dirname, threshold, subject_id, cohort]
    #np.save('/Users/jaime/vallecas/data/converters_y1/converters/results/conv_params_plot', conv_params_plot)
    afmri.plot_seed_based_correlation(wisemean_fisher, non_seed_masker_list[0], seed_coords, dirname, threshold, subject_id, cohort)
    
pdb.set_trace()               



print "Calculating the Covariance and the Precision Matrix (inverse covariance)"
precision_matrix = afmri.build_sparse_invariance_matrix(time_series, label_map)
what_to_plot = OrderedDict([('plot_heatmap', True), ('plot_graph', True), ('plot_connectome',True)])
edge_threshold = '90%'# 0.6 #'60%'
msgtitle = "Precision matrix:%s, edge threshold=%s" % (cohort,edge_threshold) 
print "Plotting the Precision Matrix (inverse covariance)"
afmri.plot_correlation_matrix(precision_matrix,label_map, what_to_plot, edge_threshold, msgtitle)

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


#Group ICA analysis
print('Calling to group ICA...')
#pdb.set_trace()
afmri.group_ICA(epi_file_list, preproc_parameters_list, cohort)

print('Calling to group Ward clustering...')
afmri.clustering_Ward(epi_file_list, preproc_parameters_list, cohort)