#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 11:25:08 2017

@author: jaime
The test_fmri.py is a module that tests the Module analysis_fmri.py 
containing preprocessing and anaylis of fMRI images
"""
import os
import sys
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
    epi_file_list_one = ['/Users/jaime/vallecas/data/surrogate_bcpa/bcpa0537_0/testwbold.nii']
    
    if cohort_group is 'converter':               
        file_list = epi_file_list_conv
        #retunr just 2 
    elif cohort_group is 'control':
        file_list = epi_file_list_control
    elif cohort_group is 'single_subject':
        #load a single image
        file_list = epi_file_list_one
    return file_list


def plot_time_series(timeseries, msgtitle=None):
    ''' plot time series, plot one time series'''
    fig1 = plt.figure()
    plt.plot(timeseries)
    if msgtitle is None:
        msgtitle = "Time series SubjectId:{}".format(msgtitle)
    plt.title(msgtitle)
    plt.xlabel('Scan number')
    plt.ylabel('Normalized signal')
    
def motion_correction(epi_file_list, preproc_parameters_list):
    ''' motion_correction: motion correction using the FSL wrapper MCFLIRT
    Input: epi_file_list path of epi images
    preproc_parameters_list: preprocessign parameters
    Output: boolean. True MCFLIRT run with no errors  '''
    moc_res  = True
    for f in epi_file_list:
        moc_res = afmri.motion_correction(f, preproc_parameters_list)
        moc_res = moc_res and moc_res
    return moc_res         
def slicetiming_correction(epi_file_list, preproc_parameters_list):
    ''' slicetiming_correction :  corrects each voxel's time-series for the fact 
    that later processing assumes that all slices were acquired exactly 
    half-way through the relevant volume's acquisition time (TR), 
    whereas in fact each slice is taken at slightly different times.
    Input: epi_file_list list of path of epi images
    Output: boolean. True if SliceTimer run with no errors  '''
    slicetiming_res = True
    for f in file_list:
        slicetiming_res = afmri.slicetiming_correction(f, preproc_parameters_list)
        slicetiming_res = slicetiming_res and slicetiming_res
    return slicetiming_res
            
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
    
def calculate_and_plot_seed_based_correlation(time_series, nonseed_masker, nonseed_ts, mask_type,mask_label,preproc_parameters_list,epi_file,seed_coords,seed_id, dirname, cohort):
    ''' calculate_seed_based_correlation and plot the contrast in MNI for one subject'''
    print "Calculating seed based correlation: one Seed vs. Entire Brain"
    # seed_ts dimension is timepints x nb of seeds (120x1)
    seed_ts = time_series[:,seed_id].reshape(time_series.shape[0],1) 
    # generate brain-wide masker from fMRI epi_file[subjectr_id]
    #nonseed_masker = afmri.generate_mask('brain-wide', [], preproc_parameters_list)
    #nonseed_ts = afmri.extract_timeseries_from_mask(nonseed_masker, epi_file_list) 
    #nonseed_ts = afmri.extract_timeseries_from_mask(nonseed_masker, epi_file) 
    # compute the seed based correlation between seed and nonseed time series per each subject
    [seed_corr_fisher,seed_corr]  = afmri.build_seed_based_correlation(seed_ts, nonseed_ts, preproc_parameters_list)
    # threshold is considered in abs value
    threshold = np.mean(seed_corr) + np.std(seed_corr)
    threshold = np.mean(seed_corr_fisher)
    # plot via inverse transform the correlation. We can plot r ot Fisher transform
    afmri.plot_seed_based_correlation_MNI_space(seed_corr_fisher, nonseed_masker, seed_coords, dirname, threshold, subject_id, cohort)
    return [seed_corr_fisher,nonseed_masker,nonseed_ts]
 
def calculate_and_plot_seed_based_coherence(time_series, nonseed_masker, nonseed_ts, mask_type, mask_label,preproc_parameters_list,epi_file,seed_coords,seed_id, dirname, cohort, freqband, typeofcorr=None):
    ''' calculate_and_plot_seed_based_coherence '''
    import pprint
    print "Calculting seed based coherence: one Seed vs. Entire Brain"
    seed_ts = time_series[:,seed_id].reshape(time_series.shape[0],1) 
    Cxy_targets, f, maskfreqs, Cxymean = calculate_seed_based_coherence(seed_ts, nonseed_ts, freqband, preproc_parameters_list)
    plot_coherence_periodogram = False
    if plot_coherence_periodogram is True:
        plot_coherence_with_seed(Cxy_targets, f, maskfreqs)
    #afmri.build_seed_based_coherence(seed_ts, nonseed_ts, preproc_parameters_list)
    threshold = 0.6
    # plot via inverse trnasform the correlation
    display = afmri.plot_seed_based_coherence_MNI_space(Cxymean, nonseed_masker, seed_coords, dirname, threshold,subject_id, cohort)
    return Cxy_targets, f, Cxymean
    
    
def calculate_seed_based_coherence(seed_ts, nonseed_ts, freqband, preproc_parameters_list):  
    ''' calculate_seed_based_coherence '''
    nb_voxels = nonseed_ts.shape[1]
    #targetseed1 = 0
    #targetseed2 = nb_voxels     
    #if all_targets == False:
    #    targetseed1 = 10345
    #    targetseed2 = 10501
    #targetseeds = targetseed2 - targetseed1
        
    nonseed_forcoh = nonseed_ts.T[0:nb_voxels,:].reshape(nb_voxels,120)
    # Estimate the magnitude squared coherence estimate, Cxy, of discrete-time signals seed and target using Welch’s method.
    Cxy_targets = []
    Cxymean = []
    passed_once = False
    for targetix in range(0,nb_voxels):
        f, Cxy = afmri.calculate_coherence(seed_ts.T, nonseed_forcoh[targetix],preproc_parameters_list)
        # we only need it once
        if passed_once is False:
            maskfreqs = (f >= freqband[0]) & (f <= freqband[1])
            passed_once = True
        Cxy = Cxy[0][maskfreqs]
        print "Mean Coherence (CPD) in range (0.01-0.1)Hz between seed and target:{} = {}".format(targetix, np.mean(Cxy))
        Cxy_targets.append(Cxy)
        Cxymean.append(np.mean(Cxy))
    Cxymean = np.asarray(Cxymean)
    Cxymean = Cxymean.reshape(nb_voxels,1)
    return Cxy_targets, f, maskfreqs, Cxymean

def plot_coherence_with_seed(Cxy, f, maskfreqs): 
    ''' plot_coherence_voxels
    Input: Cxy list of coherence between the seed and the target
    f: frequency list
    maskfreqs: subrange of f of frequencies''' 
    print "Plotting the coherence between the seed and the target, total number of plots={}".format(len(Cxy))
    for i in range(0, len(Cxy)):
        plt.semilogy(f[maskfreqs], Cxy[i])
    plt.xlabel('frequency [Hz]')
    plt.ylabel('Coherence')    
    msgtitle = "Coherence seed-target, freq band=[%s, %s]" % (freqband[0],freqband[1])      
    plt.title(msgtitle)      
    
def  load_time_series(seed_path, nonseed_path):
    ''' load_time_series load array saved repviously in file with time series seed and non seed '''
    seed_ts = np.load(seed_path)
    print "Seed ts dimensions={} X {}".format(seed_ts.shape[0], seed_ts.shape[1])
    nonseed_ts = np.load(nonseed_path)
    print "NonSeed ts dimensions={} X {}".format(nonseed_ts.shape[0], nonseed_ts.shape[1])    
    return seed_ts,nonseed_ts


#######################    
#### MAIN  PROGRAM ####
#######################
# Load preprocessing parameters   
preproc_parameters_list = preproc_parameters_list()
freqband = [0.01, 0.1]    
 
########### Loading array to do not read from File #########
load_from_file = False
if load_from_file is True:
    print " Loading time series from from array... "
    seed_ts,nonseed_ts = load_time_series('seed_ts.npy', 'nonseed_ts.npy')
    pdb.set_trace()

    
### Testing seed based coherence ######
#######################################
#all_targets = True
#print " Calculating Seed based Coherence, frequency range= {}_{}, all targets={}".format(freqband[0], freqband[1], all_targets)

#Cxy, f, maskfreqs = calculate_seed_based_coherence(seed_ts, nonseed_ts, freqband, preproc_parameters_list, all_targets=all_targets)
#maskfreqs = (f>=freqband[0]) & (f <=freqband[1])
# plot the coherece for some of the target voxels only Cxy[0:len(Cxy)]
#plot_coherence_with_seed(Cxy[0:len(Cxy)], f, maskfreqs)
#######################################
### Testing seed based coherence ######


group = ['converter', 'control', 'single_subject']
cohort = group[2]
file_list = select_cohort(cohort)

# verify and load from the full path of each images verify_and_load_images(load_list_of_epi_images(cohort))
# or using the subjects id for a hierarchical std structure: verify_and_load_images(None, '['bcpa0537_1','bcpa0578_1', 'bcpa0650_1']','/Users/jaime/vallecas/mario_fa/RF_off','wbold_data.nii')
epi_file_list = verify_and_load_images(file_list)
# Get only 2 subjects for speed
#epi_file_list = epi_file_list[3:10]
if len(epi_file_list) > 0:
    print('Nifti images for cohort:', cohort, ' loaded')
    print(epi_file_list)
    nb_of_subjects = len(epi_file_list)
    print "Nb of subjects =%s" % nb_of_subjects
    dirname = os.path.dirname(epi_file_list[0])
else:
    print('ERROR: File(s) containing the images do not exist')

# Preprocessing the images
preprocessing = True
if preprocessing is True:
    if motion_correction(epi_file_list, preproc_parameters_list) is not True:
        sys.exit("ERROR performing Motion Correction!!!!")
    pdb.set_trace()
    if slicetiming_correction(epi_file_list, preproc_parameters_list) is not True: 
        sys.exit("ERROR performing slice time correction!!!!")
               
# seed mask. for entire-brain seed based analysis the mask and the nonseed time series is genrated after
mask_type = ['atlas','DMN', 'AN', 'SN']
idx = 1

if mask_type[idx] == 'atlas':
    mask_label = 'cort-maxprob-thr25-2mm'
    #mask_label = 'power_2011'
    label_map = afmri.get_atlas_labels(mask_label)
elif (mask_type[idx] == 'DMN') or (mask_type[idx] == 'AN') or (mask_type[idx] == 'SN'):
    mask_label= afmri.get_MNI_coordinates(mask_type[idx])   
    label_map = [mask_label.keys(), mask_label.values()]
    #label_map = [afmri.get_MNI_coordinates(mask_label.keys(), afmri.get_MNI_coordinates(mask_type[idx]).values()]   
    #coords_map = afmri.get_MNI_coordinates(mask_type[idx]).values()  
    
#mask for the seed, the mask and time series for the non seed is only created if we do seed based analysis
masker = afmri.generate_mask(mask_type[idx], mask_label, preproc_parameters_list)
#PCC in DMN
seed_id = 0
seed_coords = label_map[1][seed_id]
print "The masker parameters are:%s\n" % (masker.get_params())
print "The seed = {} and its coordinates ={} \n".format(seed_id, seed_coords)
#extract time series for the masker with the preproc parameters
  

# single subject or group analysis
single_subject = True
# Seed based analysis, if False, network based analysis
pdb.set_trace()
# Extract time series from image
if single_subject is True:
    # Choose only one subject
    subject_id = 0
    time_series = afmri.extract_timeseries_from_mask(masker, epi_file_list[subject_id]) 
     # time x n voxels (eg 4 nodes in the DMN)
    seed_ts = time_series
    print "\n EXTRACTED Seed Time Series. Number of time points: {} x Voxels:{}".format(seed_ts.shape[0],seed_ts.shape[1])
    #plot only some time series
    #ts_to_plot = range(0,10)
    #plot_time_series(time_series[subject_id][:,ts_to_plot], subject_id)
    msgtitle = "DMN time series in cadaver".format(subject_id)
    plot_time_series(seed_ts, msgtitle)
    afmri.fourier_spectral_estimation(seed_ts.T, preproc_parameters_list)
else:
    # list of images extract the time series for each image
    time_series_list = []
    for i in range(0, len(epi_file_list)):
            print('........Extracting image %d / %d', (i,len(epi_file_list)-1))
            time_series = afmri.extract_timeseries_from_mask(masker, epi_file_list[i]) 
            time_series_list.append(time_series)         
    # list of time series as an array         
    seed_ts_subjects =  np.asarray(time_series_list)
    print "\n EXTRACTED Seed Time Series. Number of Subjects: {} x time points: {} x Voxels:{}".format(seed_ts_subjects.shape[0], seed_ts_subjects.shape[1],seed_ts_subjects.shape[2])
pdb.set_trace()

seed_based = False  
if seed_based is True:
    # seed based analysis, seed time series against entire brain time series
    # extract nonseed_time series
    threshold = 0.6
    if single_subject is True:
        print "\n SINGLE SUBJECT ANALYSIS: Computing Seed based correlation and Coherence....\n" 
        # time x 1 voxel
        seed_ts = extract_seed_ts(seed_ts,seed_id)
        print "Extracting time series from the entire brain....seat tight, it will take some time....\n"
        nonseed_masker, nonseed_ts = extract_non_seed_mask_and_ts(epi_file_list[subject_id], preproc_parameters_list)
        print "Calculating the seed based correlation matrix and ploting in MNI space....\n"
        seed_corr_fisher = calculate_and_plot_seed_based_correlation(seed_ts, nonseed_masker,nonseed_ts,mask_type[idx], mask_label,preproc_parameters_list, epi_file_list[subject_id],seed_coords,seed_id,dirname, cohort)
        # seed based Coherency analysis
        print "Calculating the seed based coherency matrix and ploting in MNI space....\n"
        Cxy_targets, f, Cxymean = calculate_and_plot_seed_based_coherence(seed_ts, nonseed_masker, nonseed_ts, mask_type[idx], mask_label,                                                             preproc_parameters_list, epi_file_list[subject_id],seed_coords,seed_id,dirname, cohort, freqband, 'coherence')
    else:
        #loop for the n subjects
        print "\n GROUP ANALYSIS: Computing Seed based correlation and Coherence....\n" 
        non_seed_corr_list = []
        non_seed_coh_list = []
        non_seed_masker_list = []
        non_seed_ts_list = []
        #nb_of_subjects = 2
        for subject_id in range(0, nb_of_subjects):
            print "Extracting time series for Subject %s / %s" % (subject_id, nb_of_subjects-1)
            seed_ts = seed_ts_subjects[subject_id]
            #plot_time_series(time_series[subject_id], subject_id)
            seed_ts = extract_seed_ts(seed_ts,seed_id)
            nonseed_masker, nonseed_ts = extract_non_seed_mask_and_ts(epi_file_list[subject_id], preproc_parameters_list)     
            nonseed_corr_fisher, nonseed_masker,nonseed_ts = calculate_and_plot_seed_based_correlation(seed_ts, nonseed_masker, nonseed_ts,mask_type[idx],mask_label,preproc_parameters_list,
                                                            epi_file_list[subject_id],seed_coords,seed_id,dirname, cohort)
            Cxy_targets, f, Cxymean = calculate_and_plot_seed_based_coherence(seed_ts, nonseed_masker, nonseed_ts, mask_type[idx], mask_label,preproc_parameters_list, epi_file_list[subject_id],seed_coords,seed_id,dirname, cohort, freqband, 'coherence')
            # YS calculate_and_plot_seed_based_COHERENCE
            non_seed_masker_list.append(nonseed_masker)
            non_seed_corr_list.append(nonseed_corr_fisher)
            non_seed_coh_list.append(Cxymean)
            non_seed_ts_list.append(nonseed_ts)
        #calculate the mean of the seed based across individuals
        # mean in absolute value
        #arr_fisher_corr = np.abs(np.array(non_seed_corr_list))
        arr_fisher_corr = np.array(non_seed_corr_list)
        arr_coherence = np.array(non_seed_coh_list)
        print "Wise mean of the Fisher seed correlation across subjects. min=%s, max=%s, mean=%s and std=%s." % (arr_fisher_corr.min(), arr_fisher_corr.max(), arr_fisher_corr.mean(), arr_fisher_corr.std())
        wisemean_fisher = arr_fisher_corr.mean(axis=0)
        wisemean_coh = arr_coherence.mean(axis=0)
        voxels = wisemean_fisher.shape[0]
        wisemean_fisher = wisemean_fisher.reshape(voxels,1)
        wisemean_coh = wisemean_coh.reshape(voxels,1)
        subject_id='Mean:'
        # save in file
        #np.save('/Users/jaime/vallecas/data/converters_y1/converters/results/conv_arr_fisher_corr', arr_fisher_corr)
        #conv_params_plot= [wisemean_fisher, non_seed_masker_list[0], seed_coords, dirname, threshold, subject_id, cohort]
        #np.save('/Users/jaime/vallecas/data/converters_y1/converters/results/conv_params_plot', conv_params_plot)
        afmri.plot_seed_based_correlation_MNI_space(wisemean_fisher, non_seed_masker_list[0], seed_coords, dirname, threshold, subject_id, cohort)
        print "Wise mean of the Seed Coherence (Welch method) across subjects. min=%s, max=%s, mean=%s and std=%s." % (arr_coherence.min(), arr_coherence.max(), arr_coherence.mean(), arr_coherence.std())
        display = afmri.plot_seed_based_coherence_MNI_space(wisemean_coh, non_seed_masker_list[0], seed_coords, dirname, threshold, subject_id, cohort)

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