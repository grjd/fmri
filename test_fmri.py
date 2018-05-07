#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 11:25:08 2017

@author: jaime
The test_fmri.py is a module that tests the Module analysis_fmri.py 
containing preprocessing and anaylis of fMRI images
pdb alias ll u;;d;;l
"""
import os
import sys
sys.path.insert(0, '/Users/jaime/github/code/production/network_and_topology/')
import network_analysis as neta
import numpy as np
import pdb
import matplotlib.pyplot as plt
import pandas as pd
from random import randint

import analysis_fmri as afmri
from scipy.stats import ttest_ind
from collections import OrderedDict
from nilearn import plotting
from nilearn import datasets
from nilearn.input_data import NiftiMasker 
from nilearn.datasets import load_mni152_template
from nilearn.image import resample_to_img
from nilearn.input_data import NiftiMapsMasker

print('Calling to test_fmri to test functions in analysis_fmri')
#Load a list of images load_epi_images(epi_file_list=None, subjects_list=None, dir_name=None, f_name=None)
#The location of the nifti iamges can be input
#literally epi_file_list or suing the base directory the name of the subjects and the fname
 

        
def select_cohort(cohort_group):
    '''load a list of nifti images defined in cohort_group'''
    motioncorrection = ['/Users/jaime/vallecas/data/converters_y1/converters/0015_fMRI.nii',
                          '/Users/jaime/vallecas/data/converters_y1/converters/0242_fMRI.nii',
                          '/Users/jaime/vallecas/data/converters_y1/converters/0277_fMRI.nii',
                          '/Users/jaime/vallecas/data/converters_y1/converters/0377_fMRI.nii',
                          '/Users/jaime/vallecas/data/converters_y1/converters/0464_fMRI.nii',
                          '/Users/jaime/vallecas/data/converters_y1/converters/0637_fMRI.nii',
                          '/Users/jaime/vallecas/data/converters_y1/converters/0885_fMRI.nii',
                          '/Users/jaime/vallecas/data/converters_y1/converters/0090_fMRI.nii',
                          '/Users/jaime/vallecas/data/converters_y1/converters/0243_fMRI.nii',
                          '/Users/jaime/vallecas/data/converters_y1/converters/0290_fMRI.nii',
                          '/Users/jaime/vallecas/data/converters_y1/converters/0382_fMRI.nii',
                          '/Users/jaime/vallecas/data/converters_y1/converters/0583_fMRI.nii',
                          '/Users/jaime/vallecas/data/converters_y1/converters/0707_fMRI.nii',
                          '/Users/jaime/vallecas/data/converters_y1/converters/0938_fMRI.nii',
                          '/Users/jaime/vallecas/data/converters_y1/converters/0940_fMRI.nii',
                          '/Users/jaime/vallecas/data/converters_y1/converters/0208_fMRI.nii',
                          '/Users/jaime/vallecas/data/converters_y1/converters/0263_fMRI.nii',
                          '/Users/jaime/vallecas/data/converters_y1/converters/0372_fMRI.nii',
                          '/Users/jaime/vallecas/data/converters_y1/converters/0407_fMRI.nii',
                          '/Users/jaime/vallecas/data/converters_y1/converters/0618_fMRI.nii',
                          '/Users/jaime/vallecas/data/converters_y1/converters/0882_fMRI.nii',
                          '/Users/jaime/vallecas/data/converters_y1/converters/1021_fMRI.nii']
    motioncorrection = ['/Users/jaime/vallecas/data/converters_y1/controls/0022_fMRI.nii',
                          '/Users/jaime/vallecas/data/converters_y1/controls/0028_fMRI.nii',
                          '/Users/jaime/vallecas/data/converters_y1/controls/0049_fMRI.nii',
                          '/Users/jaime/vallecas/data/converters_y1/controls/0077_fMRI.nii',
                          '/Users/jaime/vallecas/data/converters_y1/controls/0105_fMRI.nii',
                          '/Users/jaime/vallecas/data/converters_y1/controls/0121_fMRI.nii',
                          '/Users/jaime/vallecas/data/converters_y1/controls/0124_fMRI.nii',
                          '/Users/jaime/vallecas/data/converters_y1/controls/0171_fMRI.nii',
                          '/Users/jaime/vallecas/data/converters_y1/controls/0187_fMRI.nii',
                          '/Users/jaime/vallecas/data/converters_y1/controls/0189_fMRI.nii',
                          '/Users/jaime/vallecas/data/converters_y1/controls/0312_fMRI.nii',
                          '/Users/jaime/vallecas/data/converters_y1/controls/0337_fMRI.nii',
                          '/Users/jaime/vallecas/data/converters_y1/controls/0365_fMRI.nii',
                          '/Users/jaime/vallecas/data/converters_y1/controls/0429_fMRI.nii',
                          '/Users/jaime/vallecas/data/converters_y1/controls/0450_fMRI.nii',
                          '/Users/jaime/vallecas/data/converters_y1/controls/0523_fMRI.nii',
                          '/Users/jaime/vallecas/data/converters_y1/controls/0691_fMRI.nii',
                          '/Users/jaime/vallecas/data/converters_y1/controls/0805_fMRI.nii',
                          '/Users/jaime/vallecas/data/converters_y1/controls/0832_fMRI.nii',
                          '/Users/jaime/vallecas/data/converters_y1/controls/0841_fMRI.nii',
                          '/Users/jaime/vallecas/data/converters_y1/controls/0935_fMRI.nii',
                          '/Users/jaime/vallecas/data/converters_y1/controls/1017_fMRI.nii',
                          '/Users/jaime/vallecas/data/converters_y1/controls/1116_fMRI.nii']
    epi_file_list_scdplus = ['/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_scdplus/0015_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_scdplus/0023_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_scdplus/0037_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_scdplus/0045_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_scdplus/0081_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_scdplus/0084_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_scdplus/0167_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_scdplus/0205_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_scdplus/0208_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_scdplus/0231_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_scdplus/0241_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_scdplus/0268_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_scdplus/0329_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_scdplus/0373_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_scdplus/0386_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_scdplus/0440_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_scdplus/0447_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_scdplus/0464_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_scdplus/0467_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_scdplus/0533_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_scdplus/0537_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_scdplus/0549_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_scdplus/0589_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_scdplus/0667_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_scdplus/0691_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_scdplus/0700_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_scdplus/0733_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_scdplus/0736_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_scdplus/0745_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_scdplus/0760_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_scdplus/0797_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_scdplus/0816_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_scdplus/0828_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_scdplus/0831_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_scdplus/0860_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_scdplus/0885_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_scdplus/0897_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_scdplus/0899_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_scdplus/0936_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_scdplus/0940_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_scdplus/0978_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_scdplus/1017_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_scdplus/1073_fMRI.nii',]
    epi_file_list_scd_control = ['/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_control/w0032_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_control/w0061_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_control/w0075_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_control/w0109_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_control/w0114_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_control/w0122_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_control/w0225_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_control/w0234_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_control/w0245_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_control/w0256_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_control/w0295_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_control/w0299_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_control/w0313_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_control/w0320_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_control/w0345_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_control/w0355_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_control/w0357_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_control/w0365_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_control/w0370_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_control/w0390_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_control/w0416_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_control/w0429_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_control/w0437_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_control/w0456_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_control/w0468_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_control/w0469_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_control/w0495_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_control/w0520_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_control/w0523_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_control/w0551_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_control/w0552_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_control/w0630_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_control/w0663_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_control/w0674_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_control/w0710_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_control/w0716_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_control/w0748_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_control/w0762_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_control/w0769_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_control/w0792_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_control/w0805_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_control/w0842_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_control/w0845_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_control/w0859_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_control/w0865_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_control/w0935_fMRI.nii',
                          '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_control/w0964_fMRI.nii']               
    epi_file_list_healthy = ['/Users/jaime/vallecas/data/converters_y1/controls/w0022_fMRI_mcf.nii.gz',
                             '/Users/jaime/vallecas/data/converters_y1/controls/w0028_fMRI_mcf.nii.gz',
                             '/Users/jaime/vallecas/data/converters_y1/controls/w0049_fMRI_mcf.nii.gz',
                             '/Users/jaime/vallecas/data/converters_y1/controls/w0077_fMRI_mcf.nii.gz',
                             '/Users/jaime/vallecas/data/converters_y1/controls/w0105_fMRI_mcf.nii.gz',
                             '/Users/jaime/vallecas/data/converters_y1/controls/w0121_fMRI_mcf.nii.gz',
                             '/Users/jaime/vallecas/data/converters_y1/controls/w0124_fMRI_mcf.nii.gz',
                             '/Users/jaime/vallecas/data/converters_y1/controls/w0171_fMRI_mcf.nii.gz',
                             '/Users/jaime/vallecas/data/converters_y1/controls/w0187_fMRI_mcf.nii.gz',
                             '/Users/jaime/vallecas/data/converters_y1/controls/w0189_fMRI_mcf.nii.gz',
                             '/Users/jaime/vallecas/data/converters_y1/controls/w0312_fMRI_mcf.nii.gz',
                             '/Users/jaime/vallecas/data/converters_y1/controls/w0337_fMRI_mcf.nii.gz',
                             '/Users/jaime/vallecas/data/converters_y1/controls/w0365_fMRI_mcf.nii.gz',
                             '/Users/jaime/vallecas/data/converters_y1/controls/w0429_fMRI_mcf.nii.gz',
                             '/Users/jaime/vallecas/data/converters_y1/controls/w0450_fMRI_mcf.nii.gz',
                             '/Users/jaime/vallecas/data/converters_y1/controls/w0523_fMRI_mcf.nii.gz',
                             '/Users/jaime/vallecas/data/converters_y1/controls/w0691_fMRI_mcf.nii.gz',
                             '/Users/jaime/vallecas/data/converters_y1/controls/w0805_fMRI_mcf.nii.gz',
                             '/Users/jaime/vallecas/data/converters_y1/controls/w0832_fMRI_mcf.nii.gz',
                             '/Users/jaime/vallecas/data/converters_y1/controls/w0841_fMRI_mcf.nii.gz',
                             '/Users/jaime/vallecas/data/converters_y1/controls/w0935_fMRI_mcf.nii.gz',
                             '/Users/jaime/vallecas/data/converters_y1/controls/w1017_fMRI_mcf.nii.gz',
                             '/Users/jaime/vallecas/data/converters_y1/controls/w1116_fMRI_mcf.nii.gz']
    epi_file_list_conv = ['/Users/jaime/vallecas/data/converters_y1/converters/w0015_fMRI_mcf.nii.gz',
                          '/Users/jaime/vallecas/data/converters_y1/converters/w0090_fMRI_mcf.nii.gz',
                             '/Users/jaime/vallecas/data/converters_y1/converters/w0208_fMRI_mcf.nii.gz',
                             '/Users/jaime/vallecas/data/converters_y1/converters/w0242_fMRI_mcf.nii.gz',
                             '/Users/jaime/vallecas/data/converters_y1/converters/w0243_fMRI_mcf.nii.gz',
                             '/Users/jaime/vallecas/data/converters_y1/converters/w0263_fMRI_mcf.nii.gz',
                             '/Users/jaime/vallecas/data/converters_y1/converters/w0277_fMRI_mcf.nii.gz',
                             '/Users/jaime/vallecas/data/converters_y1/converters/w0290_fMRI_mcf.nii.gz',
                             '/Users/jaime/vallecas/data/converters_y1/converters/w0372_fMRI_mcf.nii.gz',
                             '/Users/jaime/vallecas/data/converters_y1/converters/w0377_fMRI_mcf.nii.gz',
                             '/Users/jaime/vallecas/data/converters_y1/converters/w0382_fMRI_mcf.nii.gz',
                             '/Users/jaime/vallecas/data/converters_y1/converters/w0407_fMRI_mcf.nii.gz',
                             '/Users/jaime/vallecas/data/converters_y1/converters/w0464_fMRI_mcf.nii.gz',
                             '/Users/jaime/vallecas/data/converters_y1/converters/w0583_fMRI_mcf.nii.gz',
                             '/Users/jaime/vallecas/data/converters_y1/converters/w0618_fMRI_mcf.nii.gz',
                             '/Users/jaime/vallecas/data/converters_y1/converters/w0637_fMRI_mcf.nii.gz',
                             '/Users/jaime/vallecas/data/converters_y1/converters/w0707_fMRI_mcf.nii.gz',
                             '/Users/jaime/vallecas/data/converters_y1/converters/w0882_fMRI_mcf.nii.gz',
                             '/Users/jaime/vallecas/data/converters_y1/converters/w0885_fMRI_mcf.nii.gz',
                             '/Users/jaime/vallecas/data/converters_y1/converters/w0938_fMRI_mcf.nii.gz',
                             '/Users/jaime/vallecas/data/converters_y1/converters/w0940_fMRI_mcf.nii.gz',
                             '/Users/jaime/vallecas/data/converters_y1/converters/w1021_fMRI_mcf.nii.gz']
    epi_file_list_one = ['/Users/jaime/vallecas/data/converters_y1/controls/w0022_fMRI.nii']
    just_testing = ['/Users/jaime/Downloads/testmni/w0022_fMRI_mcf.nii.gz', '/Users/jaime/Downloads/testmni/w0028_fMRI_mcf.nii.gz']
    
    if cohort_group is 'converter':               
        return epi_file_list_conv
        #retunr just 2 
    elif cohort_group is 'control':
        return epi_file_list_healthy
    elif cohort_group is 'single_subject':
        #load a single image
        return epi_file_list_one
    elif cohort_group is 'scdplus':
        return epi_file_list_scdplus
    elif cohort_group is 'scdcontrol':
        return epi_file_list_scd_control
    elif cohort_group is 'test':
        return just_testing
    elif cohort_group is 'motioncorrection':
      return motioncorrection

def prepare_plot_time_series(time_series, subject_id):
    """ prepare_plot_time_series: select voxels and title to plot time series """
    # plot 5 random voxels
    n_rand_voxels = [np.random.randint(0,time_series.shape[1]) for r in xrange(5)] 
    #plot specific voxels ts_to_plot = range(0,10) or just plot as such for example for DMN mask
    print "Plotting time series series voxels... \n"
    #msgtitle = "Time series in subject:{} voxels:{}".format(subject_id, n_rand_voxels)
    msgtitle = "Time series in subject:{}, DMN voxels".format(subject_id)
    #plotted_ts = time_series[:,n_rand_voxels]
    plot_time_series(time_series, msgtitle)
    plotted_ts = time_series
    #plot_time_series(plotted_ts, msgtitle)
    return plotted_ts

def plot_time_series(timeseries, msgtitle=None):
    ''' plot time series, plot one time series'''
    fig1 = plt.figure()
    plt.plot(timeseries)
    if msgtitle is None:
        msgtitle = "Time series SubjectId:{}".format(msgtitle)
    plt.title(msgtitle)
    plt.xlabel('Scan number')
    plt.ylabel('Normalized signal')
    plt.legend(loc='best')
    #figsdirectory = '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_control/figures' 
    #filenametosave = os.path.join(figsdirectory,"ts_" + msgtitle[msgtitle.find("subject"):msgtitle.find(",")])
    figsdirectory = os.getcwd()
    figsdirectory = os.path.join(figsdirectory, 'figures')
    filenametosave = os.path.join(figsdirectory,"ts_" + msgtitle[msgtitle.find("subject"):msgtitle.find(",")] +'.png')

    if not os.path.exists(figsdirectory) is True:
      os.makedirs(figsdirectory)
    
    print("Saving time series at {} \n",filenametosave)
    plt.savefig(filenametosave, bbox_inches="tight")
      
    plt.close() 

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
def read_motioncorrection_report(epi_file_list, directory=None):
    '''read_motioncorrection_report : reads mcf_results/id_report.txt files 
    Args: None by default loos at ./mcf_results
    Output:None print out the number of spikes for each image
    '''
    import re
    nusubjects = len(epi_file_list)
    dirname =  os.path.dirname(epi_file_list[0])
    reportfilefinal = os.path.join(os.path.join(dirname, 'mcf_results'), 'ReportfileFinal.txt')
    ffinal = open(reportfilefinal, 'w' )
    for i in range(0,nusubjects):
      basename = os.path.basename(epi_file_list[i])
      #get only the digits
      basenamedigits = re.findall(r'\d+', basename)[0]
      reportfile = basenamedigits + '_report.txt'
      reportfile = os.path.join(os.path.join(dirname, 'mcf_results'), reportfile)
      f = open( reportfile, 'r' )
      for line in f:
        if re.match("Found", line):
          linefinalreport = 'Subject:' + basename + '\t Report:' + line + '\n'
          print('Subject:{} \n, Report:{}\n',basename, line)
          ffinal.write(linefinalreport) 
      f.close()
    ffinal.close()

def slicetime_correction(epi_file_list, preproc_parameters_list):
    ''' slicetime_correction :  corrects each voxel's time-series for the fact 
    that later processing assumes that all slices were acquired exactly 
    half-way through the relevant volume's acquisition time (TR), 
    whereas in fact each slice is taken at slightly different times.
    Input: epi_file_list list of path of epi images
    Output: boolean. True if SliceTimer run with no errors  '''
    slicetiming_res = True
    for f in epi_file_list:
        slicetiming_res = afmri.slicetime_correction(f, preproc_parameters_list)
        slicetiming_res = slicetiming_res and slicetiming_res
    return slicetiming_res
            
def preproc_parameters_list():
    ''' Load preprocessign paarmeters as a dictionary  
    preproc_list.keys()
    preproc_list.values()'''
    preproc_list = {'standardize': 'True', 'detrend': 'True', 'smoothing_fwhm': 8, 't_r':2.5,
                    'low_pass': 0.1, 'high_pass': 0.01, 'fs': 0.4, 'nfft':129 , 'verbose':2}
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
                raise RuntimeError('ERROR .nii {} file not found!!!', f)
                epi_file_list = []
                break
            else:
                print('Image File{} found \n', f)
    else:
        #build path the list of nifti images from each subject in dirname/[subjects_list]
        if (int(subjects_list is None) + int(dir_name is None)+ int(f_name is None) <1):
            for i in range(0,len(subjects_list)):
                subjname = os.path.join(dir_name,subjects_list[i])
                print('Image File{} found \n', subjname)
                epi_file_list.append(os.path.join(subjname, f_name)) 
        else:
            raise RuntimeError('ERROR .nii {} file not found!!!', file_list[i])
            epi_file_list = [] 
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


    

def prepare_timeseries_extraction(masker, epi_file_list, subject_id=None): 
    """ prepare for extracting the timer serioes"""
    plotted_ts = []
    toplot = True
    basename = os.path.basename(epi_file_list[0])
    if subject_id is not None: #single subject
        print "Extracting time series for 1 subject \n"
        time_series = afmri.extract_timeseries_from_mask(masker, epi_file_list[subject_id]) #epi_file_list[subject_id]) 
         # time x n voxels (eg 4 nodes in the DMN)
        seed_ts = time_series
        # reshape subjects x time x voxels
        seed_ts_subjects = seed_ts.reshape(1, seed_ts.shape[0], seed_ts.shape[1])
        print "\n EXTRACTED Seed Time Series. Number of time points: {} x Voxels:{} \n".format(seed_ts.shape[0],seed_ts.shape[1])
        #plot only some time series
        plotted_ts = prepare_plot_time_series(seed_ts, subject_id)
    else:
        # list of images extract the time series for each image
        time_series_list = []
        plotted_ts_list = []
        for i in range(0, len(epi_file_list)):
                print('........Extracting image %d / %d', (i,len(epi_file_list)-1))
                time_series = afmri.extract_timeseries_from_mask(masker, epi_file_list[i]) 
                time_series_list.append(time_series)  
                plotted_ts = prepare_plot_time_series(time_series, i)
                plotted_ts_list.append(plotted_ts)    
        # list of time series as an array         
        # print('AQUI')
        # pdb.set_trace()
        seed_ts_subjects =  np.asarray(time_series_list)
        plotted_ts = np.asarray(plotted_ts_list) 
        print "\n EXTRACTED Seed Time Series. Number of Subjects: {} x time points: {} x Voxels:{}".format(seed_ts_subjects.shape[0], seed_ts_subjects[0].shape[0], seed_ts_subjects[0].shape[1])

    return seed_ts_subjects, plotted_ts


# def compute_seed_based_ttest_groups(epi_list1, epi_list2, pre_params):
#   nb_of_subjects1, nb_of_subjects2 = len(epi_list1), len(epi_list2)
#   list_stat_map1,  list_stat_map2 = [], []
#   for subject_id in range(0, nb_of_subjects1):
#     nonseed_masker, nonseed_ts = extract_non_seed_mask_and_ts(epi_list1[subject_id], pre_params)
#     stat_map = afmri.calculate_seed_based_correlation_destriaux(nonseed_ts.T, nonseed_masker)
#     #stat_map = calculate_and_plot_seed_based_correlation(time_series, nonseed_masker, nonseed_ts, mask_type, mask_label, preproc_parameters_list, epi_file, seed_coords, seed_id, dirname, cohort, subject_id):
#     list_stat_map1.append(stat_map)
#   print('Calculating stat_map for Group 1, subject:{} \n\n', subject_id)

#   for subject_id in range(0, nb_of_subjects2):
#     nonseed_masker, nonseed_ts = extract_non_seed_mask_and_ts(epi_list2[subject_id], pre_params)
#     stat_map = afmri.calculate_seed_based_correlation_destriaux(nonseed_ts.T, nonseed_masker)
#     list_stat_map2.append(stat_map)
#   print('Calculating stat_map for Group 2, subject:{} \n\n', subject_id)
#   #ttest 
#   stat_map_g1 = pd.DataFrame(list_stat_map1)
#   stat_map_g2 = pd.DataFrame(list_stat_map2)
#   return stat_map1, stat_map2

def compute_network_based_analysis(matrices, labels):
  """ compute_network_based_analysis 
  Args:matrices list of matrices or single matrix
  Output: 
  """
  #convert matrix into a dataframe 
  if isinstance(matrices, (list,)): 
    print('Loop for the list of matrices')
    for i in matrices:
      curmat = matrices
      #convert ndarray into dataframe
      df = pd.DataFrame(curmat,labels, labels)
      print("Labels of the Connectivity Matrix : " + str(df.columns.values))
      G = neta.build_graph_correlation_matrix(df, threshold=None)


def plot_graph_from_atlas(corr_matrix, time_series, atlas):
  """plot_graph_from_atlas  plot the corresponding graph from an atlas (extratcted time series)
  Args: corr_matrix squared matrix, time_series: time series 
  Output """

#######################  ####################### ####################### #######################  
#### MAIN  PROGRAM ####
####################### ####################### ####################### #######################
def main():
    plt.close('all')
    # Load preprocessing parameters   
    pre_params = preproc_parameters_list()
    freqband = [0.01, 0.1]    
    freqband = [pre_params['high_pass'], pre_params['low_pass']]
    #nonseed_mask = afmri.generate_mask('brain-wide', pre_params)
    ################################
    # mcf motion correction
    # stc slice tiem correction
    ################################ 

    mcf = False
    stc  = False
    if mcf is True:
        print('Performing Motion Correction....\n)')
        cohort = 'motioncorrection'
        file_list = select_cohort(cohort)
        # verify and load from the full path of each images verify_and_load_images(load_list_of_epi_images(cohort))
        # or using the subjects id for a hierarchical std structure: verify_and_load_images(None, '['bcpa0537_1','bcpa0578_1', 'bcpa0650_1']','/Users/jaime/vallecas/mario_fa/RF_off','wbold_data.nii')
        epi_file_list = verify_and_load_images(file_list)
        print('Images found at:{}', epi_file_list)
        basename = os.path.basename(epi_file_list[0])
        dirname = os.path.dirname(epi_file_list[0])
        #change current directory
        print('Changing directory to:{}',dirname)
        os.chdir(dirname)
        print('Changed directory to:{}',os.getcwd())
        if motion_correction(epi_file_list, pre_params) is not True:
            sys.exit("ERROR performing Motion Correction!")
        read_motioncorrection_report(epi_file_list)
        print('Motion correction Finished. Perform MNI normalization to continue \n')
        #print('go to /Users/jaime/github/code/production/matlab-normalization/normalize-boldMNI-vPython.m') 
        print('When MNI normalization is done set file_list and mcf = False :) \n')
        #3 Steps: 1.gzip -kd *.nii.gz
        #         2. Normalize
        #         3. gzip w*.nii
        # rename epi_file_list adding _mcf
        # epi_file_list_mcf = list()
        # for i in range(0,len(epi_file_list)):
        #     base = os.path.splitext(epi_file_list[i])[0]
        #     base = base + '_mcf.nii.gz'
        #     epi_file_list_mcf.append(base)
        # epi_file_list = verify_and_load_images(epi_file_list_mcf)
        sys.exit()
    
    if stc is True:
        if slicetime_correction(epi_file_list, pre_params) is not True: 
            sys.exit("ERROR performing Slice time correction!") 
        print('Slice Time Correction Finished. Perform MNI normalization to continue \n')
        print('When MNI normalization is done set file_list and stc = False \n')
        sys.exit()
    
    ########### Loading array to do not read from File #########
    # load_from_file = False
    # if load_from_file is True:
    #     print " Loading time series from from array... "
    #     seed_ts,nonseed_ts = load_time_series('seed_ts.npy', 'nonseed_ts.npy')

    # load and verify from the full path of each images
    group = ['converter', 'control', 'single_subject', 'scdplus', 'scdcontrol', 'motioncorrection', 'test']
    cohort = group[-1]
    file_list = select_cohort(cohort)
    epi_file_list = verify_and_load_images(file_list)
    print('Images found at:{}', epi_file_list)
    basename = os.path.basename(epi_file_list[0])
    dirname = os.path.dirname(epi_file_list[0])
    #change current directory
    print('Changing directory to:{}',dirname)
    os.chdir(dirname)
    print('Changed directory to:{}',os.getcwd())
          
    #######################################
    # Create mask from which to extract   #
    # time series                         #
    #                                     #             
    #######################################
    
    mask_type = ['atlas-msdl','cort-maxprob-thr25-2mm', 'sub-maxprob-thr25-2mm', 'DMN']#, 'AN', 'SN', 'brain-wide']
    mask_type = mask_type[0]
    if mask_type.find('DMN') > -1 :
      print('The Mask type is a Network not an Atlas...\n')
      mask_label = afmri.get_MNI_coordinates(mask_type)
      labels = mask_label.keys()
      coords = mask_label.values()
      print('labels are {} coords are {} \n',labels, coords)
    elif mask_type.find('msdl') > -1:
      atlas = datasets.fetch_atlas_msdl()
      atlas_filename = atlas['maps']
      labels = atlas['labels']
      coords = atlas.region_coords
    elif mask_type.find('cort-maxprob-thr25-2mm') > -1:
      atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
      atlas_filename = atlas.maps
      labels = atlas.labels
      #coords are missing!!

    elif mask_type.find('sub-maxprob-thr25-2mm') > -1:
      atlas = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
      atlas_filename = atlas.maps
      labels = atlas.labels
      #coords are missing!!

    print("\n Calling to Generate Mask type:" + mask_type + "\n")  
    masker = afmri.generate_mask(mask_type, pre_params, epi_filename=None)
    #######################################
    # Extract time series from the mask   #
    #                                     #
    ####################################### 
    #signal is clean in afmri.extract_timeseries_from_mask to increase SNR
    seed_ts, plotted_ts = prepare_timeseries_extraction(masker, epi_file_list)#, subject_id=0)
    # seed_ts == plotted_ts when more than 1 subject
    seed_ts_subjects = seed_ts
    # remove 4 first slices
    #seed_ts_subjects[:,4:,:], plotted_ts[:,4:,:]
    
    #######################################
    # fourier_spectral_estimation.        #
    #######################################
    #plot spectra for all subjects or only one
    #Set in fourier_spectral_estimation the variable figsdirectory with the directory where the plots will be saved
    #figsdirectory = '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_scdplus/figures/'
    #YS: FIX to calculate the PSD also for Atlas not always DMN
    psd_allsubjects = list()
    for i in range(0, seed_ts.shape[0]):
        print("Calculating PSD for subject:{}/{}", i,seed_ts.shape[0])
        psd = afmri.fourier_spectral_estimation(seed_ts_subjects[i].T, pre_params, 'subject:' + str(i))
        psd_allsubjects.append(psd)
  

    #######################################
    # Seed based analysis                 #
    # Pearson correlation (power based)   #
    # and coherence    Mask=DMN           #
    ####################################### 
    seed_based = False
    seed_id = 0 # PCC in the DMN
    if seed_based is True:
      print('\n\n Calling to build_seed_based_stat_map mask type must be DMN. GROUP 1' )
      list_corr_stat_map, list_coh_stat_map = [], []
      epi_file_list1 = verify_and_load_images(select_cohort('converter'))
      dirname =  os.path.dirname(epi_file_list1[0])
      #change current directory
      #print('Changing directory to:{}',dirname)
      #os.chdir(dirname)
      #print('Changed directory to:{}',os.getcwd())
      seed_ts_subjects, plotted_ts = prepare_timeseries_extraction(masker, epi_file_list1)
      #seed_ts_subjects = seed_ts_subjects[:,4:,:]
      corr_stat_map, coh_stat_map, nonseed_masker = afmri.build_seed_based_stat_map(epi_file_list1, seed_ts_subjects, pre_params, mask_type, coords, seed_id, dirname, cohort)
      list_corr_stat_map.append(corr_stat_map)
      list_coh_stat_map.append(coh_stat_map)
      # Get the stat map for another group oin order to study difference
      print('\n\n Calling to build_seed_based_stat_map mask type must be DMN. GROUP 2' )
      epi_file_list2 = verify_and_load_images(select_cohort('control'))
      dirname =  os.path.dirname(epi_file_list2[0])
      seed_ts_subjects, plotted_ts = prepare_timeseries_extraction(masker, epi_file_list2)
      #seed_ts_subjects = seed_ts_subjects[:,4:,:]
      corr_stat_map, coh_stat_map, nonseed_masker = afmri.build_seed_based_stat_map(epi_file_list2, seed_ts_subjects, pre_params, mask_type, coords, seed_id, dirname, cohort)
      list_corr_stat_map.append(corr_stat_map)
      list_coh_stat_map.append(coh_stat_map)

      print('\n.....compute_seed_based_ttest_groups for 2 groups\n')
      
      stat_map_g1 = list_corr_stat_map[0] 
      stat_map_g2 = list_corr_stat_map[1] 
      type_stat_map = 'Correlation'
      print('\n.....Correlation ttest_stat_map_groups for 2 groups\n')
      threshold = 0.6 # threshold can be blank it will be assigned in the function
      ttest_stat_map_groups, pval_stat_map_groups, pvalscorr_stat_map_groups = afmri.ttest_stat_map_groups(stat_map_g1, stat_map_g2, nonseed_masker, dirname, threshold, type_stat_map, coords[seed_id])
      
      type_stat_map = 'Coherence'
      stat_map_g1 = list_coh_stat_map[0]
      stat_map_g2 = list_coh_stat_map[1] 
      print('\n.....Coherence ttest_stat_map_groups for 2 groups\n')
      ttest_stat_map_groups, pval_stat_map_groups, pvalscorr_stat_map_groups = afmri.ttest_stat_map_groups(stat_map_g1, stat_map_g2, nonseed_masker, dirname, threshold, type_stat_map, coords[seed_id])
      print("\n Done with Seed Based Correlation=Coherence for two groups, see results at: " + str(dirname) + "/figures" + "\n\n")  
      print('Exiting Program...')
      return 0
    #######################################
    # Build connectome in time domain     # 
    # Correlation/Covariance/Precision    #
    #                                     #
    ####################################### 

    print("Building connectome in Time domain ...\n")
    kind_of_correlation = ['correlation', 'covariance', 'tangent', 'precision', 'partial correlation', 'precision_matrix_sparseCV', 'cov_matrix_sparseCV']
    #correlation and covariance return identical result
    #corr_matrices = afmri.build_connectome(seed_ts_subjects, kind_of_correlation=kind_of_correlation[0])
    cov_matrices = afmri.build_connectome(seed_ts_subjects, kind_of_correlation='covariance')
    tangent_matrices = afmri.build_connectome(seed_ts_subjects, kind_of_correlation='tangent')
    precision_matrices = afmri.build_connectome(seed_ts_subjects, kind_of_correlation='precision')
    pcorr_matrices = afmri.build_connectome(seed_ts_subjects, kind_of_correlation='partial correlation')
    # Build Group Covariance and the Precision Matrix using using GroupSparseCovarianceCV
    print "Calculating the Group Covariance and the Precision Matrix (inverse covariance) \n"
    precision_matrix_sparseCV, cov_matrix_sparseCV = afmri.build_sparse_invariance_matrix(seed_ts_subjects, labels)
    
    #save the matrices
    matricesdirectory = os.path.join(os.getcwd(), 'matrices')
    if not os.path.exists(matricesdirectory) is True:
      print('Creating matrices directory....\n')
      os.makedirs(matricesdirectory)
    
    #pdb.set_trace()
    arraytosave = os.path.join(matricesdirectory,"cov_matrices_" + mask_type + ".npy")
    print("Saving cov_matrices at {} \n",arraytosave) 
    np.save(arraytosave, cov_matrices)
    arraytosave = os.path.join(matricesdirectory,"tangent_matrices_" + mask_type + ".npy")
    print("Saving tangent_matrices at {} \n",arraytosave)
    np.save(arraytosave, tangent_matrices)
    arraytosave = os.path.join(matricesdirectory,"precision_matrices_" + mask_type + ".npy")
    print("Saving precision_matrices at {} \n",arraytosave)
    np.save(arraytosave, precision_matrices)
    arraytosave = os.path.join(matricesdirectory,"pcorr_matrices_" + mask_type + ".npy")
    print("Saving pcorr_matrices at {} \n",arraytosave)
    np.save(arraytosave, pcorr_matrices)
    arraytosave = os.path.join(matricesdirectory,"precision_matrix_sparseCV_" + mask_type + ".npy")
    print("Saving precision_matrix_sparseCV at {} \n",arraytosave)
    np.save(arraytosave, precision_matrix_sparseCV)
    arraytosave = os.path.join(matricesdirectory,"cov_matrix_sparseCV_" + mask_type + ".npy")
    print("Saving cov_matrix_sparseCV at {} \n", arraytosave)
    np.save(arraytosave, cov_matrix_sparseCV)

    #######################################
    # Build connectome in Frequency domain# 
    # Coherence                           #
    #                                     #
    ####################################### 
    frequency_analysis = True
    if frequency_analysis is True:
      kind_of_correlation = ['coherency', 'correlation', 'covariance', 'tangent', 'precision', 'partial correlation', 'precision_matrix_sparseCV', 'cov_matrix_sparseCV']
      print "Building connectome in Frequency domain. Coherency...\n"
      coherency_matrices = afmri.build_connectome_in_frequency(seed_ts_subjects, pre_params, freqband)
      # convert list into ndarray subjects x time x voxels 
      coherency_matrices =  np.asarray(coherency_matrices)
      arraytosave = os.path.join(matricesdirectory,"coherency_matrices_" + mask_type + ".npy")
      print("Saving coherency_matrices at {} \n",arraytosave) 
      np.save(arraytosave, coherency_matrices)
      # mean across subjects
      coherency_mean_subjects = coherency_matrices.mean(axis=0)
    
    #######################################
    # Plot connectome in time domian      #
    # mean for subjects and/or            #
    #      single subjects                # 
    #######################################     
    print("Plotting the connectome matrices for correlation type: " + str(kind_of_correlation) + "\n\n")
    what_to_plot = OrderedDict([('plot_heatmap', True), ('plot_graph', True)])
    
    for idcor in kind_of_correlation:
      #select the connectome_to_plot
      if idcor is 'coherency':
        connectome_to_plot = coherency_matrices
        connectome_to_plot_mean = coherency_mean_subjects
      if idcor is 'covariance':
        connectome_to_plot = cov_matrices
      elif idcor is 'tangent':
        connectome_to_plot = tangent_matrices
      elif idcor is 'precision':
        connectome_to_plot = precision_matrices
      elif idcor is 'partial correlation':
        connectome_to_plot = pcorr_matrices
      elif idcor is 'precision_matrix_sparseCV':
        connectome_to_plot = precision_matrix_sparseCV
        connectome_to_plot_mean = connectome_to_plot
      elif idcor is 'partial cov_matrix_sparseCV':
        connectome_to_plot = cov_matrix_sparseCV
        connectome_to_plot_mean = connectome_to_plot

      print('Transforming connectome_to_plot to ndarray \n')
      if type(connectome_to_plot) is list:
        connectome_to_plot = np.transpose(np.asarray(connectome_to_plot))
        connectome_to_plot_mean = connectome_to_plot.mean(-1)   
      if mask_type.find('msdl') > -1:
        print('Plotting the Mean subjects connectome for MSDL Atlas .... \n')
        plotting.plot_connectome(connectome_to_plot_mean, coords, edge_threshold="80%", colorbar=True)
        # for a single subject eg id=0
        #plotting.plot_connectome(connectome_to_plot[:,:,0], coords, edge_threshold="80%", colorbar=True)
        print('Plotting the heat map and graph for MSDL Atlas .... \n')
        msgtitle = "Mean connectome. Group:{}, Mask:{}, Corr:{}".format(cohort, mask_type, idcor)
        afmri.plot_correlation_matrix(connectome_to_plot_mean, labels, msgtitle, what_to_plot)

        print "Plotting Sparse CV Matrix (inverse covariance) \n"
        msgtitle = "Sparse CV. Group:{}, Mask:{}, Corr:{}".format(cohort, mask_type, 'sparse CV')
        afmri.plot_correlation_matrix(precision_matrix_sparseCV, labels, msgtitle, what_to_plot)
      
      elif mask_type.find('DMN') > -1:
        print('Plotting the Mean subjects connectome for the DMN .... \n')
        plotting.plot_connectome(connectome_to_plot_mean, coords, edge_threshold="80%", colorbar=True)
        # for a single subject eg id=0
        #plotting.plot_connectome(connectome_to_plot[:,:,0], coords, edge_threshold="80%", colorbar=True)
        print('Plotting the heat map and graph for the DMN .... \n')
        msgtitle = "Mean connectome. Group:{}, Mask:{}, Corr:{}".format(cohort, mask_type,idcor)
        afmri.plot_correlation_matrix(connectome_to_plot_mean, labels, msgtitle, what_to_plot)
        print "Plotting tSparse CV Matrix (inverse covariance) \n"
        msgtitle = "Sparse CV. Group:{}, Mask:{}, Corr:{}".format(cohort, mask_type, 'sparse CV')
        afmri.plot_correlation_matrix(precision_matrix_sparseCV, labels, msgtitle, what_to_plot)
      elif mask_type.find('maxprob') > -1:
        print('Plotting the heat map and graph for Harvard Oxford : {}  \n', mask_type)
        msgtitle = "Mean connectome. Group:{}, Mask:{}, Corr:{}".format(cohort, mask_type, idcor)
        afmri.plot_correlation_matrix(connectome_to_plot_mean, labels, msgtitle, OrderedDict([('plot_heatmap', True), ('plot_graph', False)]))
        print "Plotting Sparse CV Matrix (inverse covariance) \n"
        msgtitle = "Sparse CV. Group:{}, Mask:{}, Corr:{}".format(cohort, mask_type, 'sparse CV')
        afmri.plot_correlation_matrix(precision_matrix_sparseCV, labels, msgtitle, OrderedDict([('plot_heatmap', True), ('plot_graph', False)]))
        # Cant plot the connetome because I dont have the coords

    print("\n\n Connectome DONE results at: " + str(matricesdirectory) + "\n\n")
    #print('Exiting...')
    #return 0

    print('Network based analysis....')
    compute_network_based_analysis()

    #######################################
    # Granger causality                   #
    # test and plot Granger connectome    #
    #                                     #
    ####################################### 
    print "\n\nCalculating granger causality matrix, subjects:%d Mask type:%s" %(len(epi_file_list), mask_type)
    #granger_test_results = granger_causality_analysis(seed_ts_subjects[0], pre_params,label_map, order=10)
    #YS: Need the average , check doesnt work


    #######################################
    # Group ICA and Ward clustering       #
    #                                     #
    ####################################### 

    print('\n\n Calling to group ICA...\n\n\n\n')
    afmri.group_ICA(epi_file_list, pre_params, cohort)

    print('\n\n Calling to group Ward clustering...\n\n\n\n')
    afmri.clustering_Ward(epi_file_list, pre_params, cohort)
    print('\n\n ----- END of test_fmri.py --- \n\n')