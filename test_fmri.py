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
import numpy as np
import pdb
import matplotlib.pyplot as plt
import analysis_fmri as afmri
from collections import OrderedDict
from nilearn import plotting
from nilearn import datasets
from nilearn.input_data import NiftiMasker 
from nilearn.datasets import load_mni152_template
from nilearn.image import resample_to_img

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
    epi_file_list_control = ['/Users/jaime/vallecas/data/converters_y1/controls/0022_fMRI.nii',
                             '/Users/jaime/vallecas/data/converters_y1/controls/0077_fMRI.nii',
                             '/Users/jaime/vallecas/data/converters_y1/controls/0124_fMRI.nii',
                             '/Users/jaime/vallecas/data/converters_y1/controls/0189_fMRI.nii',
                             '/Users/jaime/vallecas/data/converters_y1/controls/0365_fMRI.nii',
                             '/Users/jaime/vallecas/data/converters_y1/controls/0523_fMRI.nii',
                             '/Users/jaime/vallecas/data/converters_y1/controls/0832_fMRI.nii',
                             '/Users/jaime/vallecas/data/converters_y1/controls/1017_fMRI.nii',
                             #'/Users/jaime/vallecas/data/converters_y1/controls/0243_fMRI.nii',
                             '/Users/jaime/vallecas/data/converters_y1/controls/0028_fMRI.nii',
                             '/Users/jaime/vallecas/data/converters_y1/controls/0105_fMRI.nii',
                             '/Users/jaime/vallecas/data/converters_y1/controls/0171_fMRI.nii',
                             '/Users/jaime/vallecas/data/converters_y1/controls/0312_fMRI.nii',
                             '/Users/jaime/vallecas/data/converters_y1/controls/0429_fMRI.nii',
                             '/Users/jaime/vallecas/data/converters_y1/controls/0691_fMRI.nii',
                             '/Users/jaime/vallecas/data/converters_y1/controls/0841_fMRI.nii',
                             '/Users/jaime/vallecas/data/converters_y1/controls/1116_fMRI.nii',
                             '/Users/jaime/vallecas/data/converters_y1/controls/0049_fMRI.nii',
                             '/Users/jaime/vallecas/data/converters_y1/controls/0121_fMRI.nii',
                             '/Users/jaime/vallecas/data/converters_y1/controls/0187_fMRI.nii',
                             '/Users/jaime/vallecas/data/converters_y1/controls/0337_fMRI.nii',
                             '/Users/jaime/vallecas/data/converters_y1/controls/0450_fMRI.nii',
                             '/Users/jaime/vallecas/data/converters_y1/controls/0805_fMRI.nii',
                             '/Users/jaime/vallecas/data/converters_y1/controls/0935_fMRI.nii']
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
    epi_file_list_one = ['/Users/jaime/vallecas/data/cyst_arach/wcyst_fMRI_RESTING_S_20171018163846_10.nii']
    epi_file_list_one = ['/Users/jaime/Downloads/s_008.nii']
    just_testing = ['/Users/jaime/Downloads/just_test/w0313_fMRI.nii', '/Users/jaime/Downloads/just_test/w0469_fMRI.nii']
    
    if cohort_group is 'converter':               
        return epi_file_list_conv
        #retunr just 2 
    elif cohort_group is 'control':
        return epi_file_list_control
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

def extract_non_seed_mask_and_ts(epi_file, preproc_parameters_list):    
    ''' extract_non_seed_mask_and_ts '''
    nonseed_masker = afmri.generate_mask('brain-wide', [], preproc_parameters_list, epi_file)
    nonseed_ts = afmri.extract_timeseries_from_mask(nonseed_masker, epi_file) 
    return nonseed_masker, nonseed_ts
    
def calculate_and_plot_seed_based_correlation(time_series, nonseed_masker, nonseed_ts, mask_type,mask_label,preproc_parameters_list,epi_file,seed_coords,seed_id, dirname, cohort, subject_id):
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

def calculate_and_plot_seed_based_coherence(time_series, nonseed_masker, nonseed_ts, mask_type, mask_label,preproc_parameters_list,epi_file,seed_coords,seed_id, dirname, cohort, freqband, subject_id, typeofcorr=None):
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
    display = afmri.plot_seed_based_coherence_MNI_space(Cxymean, nonseed_masker, seed_coords, dirname, threshold, subject_id, cohort)
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
        
    nonseed_forcoh = nonseed_ts.T[0:nb_voxels,:].reshape(nb_voxels,seed_ts.shape[0])
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
        print "Mean Coherence (CPD) in range {}-{} Hz. between seed and target:{} = {}".format(freqband[0],freqband[1], targetix, np.mean(Cxy))
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
    
def load_time_series(seed_path, nonseed_path):
    ''' load_time_series load array saved repviously in file with time series seed and non seed '''
    seed_ts = np.load(seed_path)
    print "Seed ts dimensions={} X {}".format(seed_ts.shape[0], seed_ts.shape[1])
    nonseed_ts = np.load(nonseed_path)
    print "NonSeed ts dimensions={} X {}".format(nonseed_ts.shape[0], nonseed_ts.shape[1])    
    return seed_ts,nonseed_ts

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
        seed_ts_subjects =  np.asarray(time_series_list)
        plotted_ts = np.asarray(plotted_ts_list) 
        print "\n EXTRACTED Seed Time Series. Number of Subjects: {} x time points: {} x Voxels:{}".format(seed_ts_subjects.shape[0], seed_ts_subjects[0].shape[0], seed_ts_subjects[0].shape[1])
    return seed_ts_subjects, plotted_ts

def prepare_mask_creation(preproc_parameters_list, seed_based, mask_type):
    """ prepare_mask_creation : returns masker to be callled to extract time series
    Args: preproc_parameters_list
    Output: masker, mask_label, label_map, seed_id, seed_coords, mask_type[idx]"""
    # seed mask. for entire-brain seed based analysis the mask and the nonseed time series is generated after

    if mask_type == 'atlas':
        mask_label = 'cort-maxprob-thr25-2mm'
        #mask_label = 'msdl'
        #mask_label = 'power_2011'
        label_map = afmri.get_atlas_labels(mask_label)

    elif (mask_type == 'DMN') or (mask_type == 'AN') or (mask_type == 'SN'):
        mask_label= afmri.get_MNI_coordinates(mask_type)   
        label_map = [mask_label.keys(), mask_label.values()]
        #label_map = [afmri.get_MNI_coordinates(mask_label.keys(), afmri.get_MNI_coordinates(mask_type[idx]).values()]   
        #coords_map = afmri.get_MNI_coordinates(mask_type[idx]).values()  

    #mask for the seed, the mask and time series for the non seed is only created if we do seed based analysis
    print('Calling to afmri.generate_mask {} {} {} \n',mask_type, mask_label, preproc_parameters_list)
    masker = afmri.generate_mask(mask_type, mask_label, preproc_parameters_list)
    seed_id = None
    seed_coords = None
    if seed_based is True:
      #PCC in DMN
      seed_id = 0
      seed_coords = label_map[1][seed_id]
      print "The masker parameters are:%s\n" % (masker.get_params())
      print "The seed = {} and its coordinates ={} \n".format(seed_id, seed_coords)
    return masker, mask_label, label_map, seed_id, seed_coords, mask_type

#######################  ####################### ####################### #######################  
#### MAIN  PROGRAM ####
####################### ####################### ####################### #######################
def main():
    plt.close('all')
    # Load preprocessing parameters   
    pre_params = preproc_parameters_list()
    freqband = [0.01, 0.1]    
    freqband = [pre_params['high_pass'], pre_params['low_pass']]
    ################################
    # mcf motion correction
    # stc slice tiem correction
    ################################ 
    group = ['converter', 'control', 'single_subject', 'scdplus', 'scdcontrol', 'motioncorrection']
    cohort = group[-1]
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

    mcf = False
    stc  = False
    if mcf is True:
        print('Performing Motion Correction....\n)')
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
    load_from_file = False
    if load_from_file is True:
        print " Loading time series from from array... "
        seed_ts,nonseed_ts = load_time_series('seed_ts.npy', 'nonseed_ts.npy')

    # verify and load from the full path of each images
    group = ['converter', 'control', 'single_subject', 'scdplus', 'scdcontrol', 'motioncorrection']
    cohort = group[0]
    file_list = select_cohort(cohort)
    epi_file_list = verify_and_load_images(file_list)
    print('Images found at:{}', epi_file_list)
    basename = os.path.basename(epi_file_list[0])
    dirname = os.path.dirname(epi_file_list[0])
    #change current directory
    print('Changing directory to:{}',dirname)
    os.chdir(dirname)
    print('Changed directory to:{}',os.getcwd())
    # Get only 2 subjects for speed
    #epi_file_list = epi_file_list[3:10]
    if len(epi_file_list) > 0:
        print('Nifti images for cohort:', cohort, ' loaded')
        print(epi_file_list)
        nb_of_subjects = len(epi_file_list)
        print "Nb of subjects = %s" % nb_of_subjects
        dirname = os.path.dirname(epi_file_list[0])
    else:
        sys.exit('ERROR: File(s) containing the images do not exist')
          
    #######################################
    # Create masker from which to extract #
    # time series.                        #
    # mask_type = ['atlas','DMN', 'AN',   #
    #'SN', 'brain_wide']                  #
    #######################################
    seed_based = True
    mask_type = ['atlas','DMN']#, 'AN', 'SN', 'brain-wide']
    mask_type = mask_type[1]
    masker, mask_label, label_map, seed_id, seed_coords, mask_type = prepare_mask_creation(pre_params, seed_based, mask_type)
    #######################################
    # Extract time series from the masker #
    # with the preproc parameters         #
    # single subject or group analysis    # 
    # subject_id is optional, if none.    #
    # process all images  
    # Example: (1 subject) prepare_timeseries_extraction(masker, epi_file_list, subject_id=0) #
    #          (Group) prepare_timeseries_extraction(masker, epi_file_list) 
    ####################################### 

    #Set in plot_time_series the vartiable figsdirectory with the directory where the plots will be saved
    #figsdirectory = '/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_scdplus/figures/'
    seed_ts, plotted_ts = prepare_timeseries_extraction(masker, epi_file_list)#, subject_id=0)
    seed_ts_subjects = seed_ts
    #prepare_timeseries_extraction(masker, epi_file_list)
    
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
        psd = afmri.fourier_spectral_estimation(seed_ts[i].T, pre_params, str(i))
        psd_allsubjects.append(psd)
    
    #YS: plot psd of the average of all subjects
    #psd = afmri.fourier_spectral_estimation(<calculate the mean subjects 120x4>, pre_params)
    #######################################
    # Seed based analysis                 #
    # Pearson correlation (power based)   #
    # and coherence                       #
    ####################################### 
    
    # threshold used when plotting the results 
    threshold = 0.6 
    single_subject = False
    subject_id = 0
    seed_id = 0
    if seed_based is True:
        if single_subject is True:
            print "\n SINGLE SUBJECT ANALYSIS: Computing Seed based correlation and Coherence....\n" 
            # time x 1 voxel
            seed_ts = extract_seed_ts(seed_ts,seed_id)
            print "Extracting time series from the entire brain....seat tight, it will take some time....\n"
            nonseed_masker, nonseed_ts = extract_non_seed_mask_and_ts(epi_file_list[subject_id], pre_params)
            print "Calculating the seed based correlation matrix and ploting in MNI space....\n"
            #seed_corr_fisher = calculate_and_plot_seed_based_correlation(seed_ts, nonseed_masker,nonseed_ts,mask_type[idx], mask_label,pre_params, epi_file_list[subject_id],seed_coords,seed_id,dirname, cohort)
            seed_corr_fisher = calculate_and_plot_seed_based_correlation(seed_ts, nonseed_masker,nonseed_ts, mask_type, mask_label,pre_params, epi_file_list[subject_id],seed_coords,seed_id,dirname, cohort, subject_id)
            # seed based Coherency analysis
            print "Calculating the seed based coherency matrix and ploting in MNI space....\n"
            Cxy_targets, f, Cxymean = calculate_and_plot_seed_based_coherence(seed_ts, nonseed_masker, nonseed_ts, mask_type, mask_label, pre_params, epi_file_list[subject_id],seed_coords,seed_id,dirname, cohort, freqband, subject_id, 'coherence')
        else:
            #loop for the n subjects
            print "\n GROUP ANALYSIS: Computing Seed based correlation and Coherence....\n" 
            non_seed_corr_list = []
            non_seed_coh_list = []
            non_seed_masker_list = []
            non_seed_ts_list = []
            #nb_of_subjects = 2
            for subject_id in range(0, nb_of_subjects):
                print "Extracting time series for Subject %s / %s \n" % (subject_id, nb_of_subjects-1)
                seed_ts = seed_ts_subjects[subject_id]
                #plot_time_series(time_series[subject_id], subject_id)
                seed_ts = extract_seed_ts(seed_ts,seed_id)
                nonseed_masker, nonseed_ts = extract_non_seed_mask_and_ts(epi_file_list[subject_id], pre_params)     
                nonseed_corr_fisher, nonseed_masker,nonseed_ts = calculate_and_plot_seed_based_correlation(seed_ts, nonseed_masker, nonseed_ts,'DMN',mask_label,preproc_parameters_list, epi_file_list[subject_id],seed_coords,seed_id,dirname, cohort,subject_id)

                Cxy_targets, f, Cxymean = calculate_and_plot_seed_based_coherence(seed_ts,nonseed_masker, nonseed_ts, 'DMN', mask_label, pre_params, epi_file_list[subject_id],seed_coords,seed_id,dirname, cohort, freqband, subject_id,'coherence')

                # YS calculate_and_plot_seed_based_COHERENCE
                non_seed_masker_list.append(nonseed_masker)
                non_seed_corr_list.append(nonseed_corr_fisher)
                non_seed_coh_list.append(Cxymean)
                non_seed_ts_list.append(nonseed_ts)
            #calculate the mean of the seed based across individuals
            #mean in absolute value
            #arr_fisher_corr = np.abs(np.array(non_seed_corr_list))
            arr_fisher_corr = np.array(non_seed_corr_list)
            arr_coherence = np.array(non_seed_coh_list)
            #print "Wise mean of the Fisher seed correlation across subjects. min=%s, max=%s, mean=%s and std=%s." % (arr_fisher_corr.min(), arr_fisher_corr.max(), arr_fisher_corr.mean(), arr_fisher_corr.std())
            wisemean_fisher = arr_fisher_corr.mean(axis=0)
            wisemean_coh = arr_coherence.mean(axis=0)
            voxels = wisemean_fisher.shape[0]
            wisemean_fisher = wisemean_fisher.reshape(voxels,1)
            wisemean_coh = wisemean_coh.reshape(voxels,1)
            subject_id='Mean:'
            # save in file
            #np.save('/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_control/figures/conv_arr_fisher_corr', arr_fisher_corr)
            np.save('figures/conv_arr_fisher_corr', arr_fisher_corr)
            conv_params_plot= [wisemean_fisher, non_seed_masker_list[0], seed_coords, dirname, threshold, subject_id, cohort]
            #np.save('/Users/jaime/vallecas/data/scc/scc_image_subjects/preprocessing/prep_control/figures/conv_params_plot', conv_params_plot)
            np.save('figures/conv_params_plot', conv_params_plot)
            afmri.plot_seed_based_correlation_MNI_space(wisemean_fisher, non_seed_masker_list[0], seed_coords, dirname, threshold, subject_id, cohort)
            #print "Wise mean of the Seed Coherence (Welch method) across subjects. min=%s, max=%s, mean=%s and std=%s." % (arr_coherence.min(), arr_coherence.max(), arr_coherence.mean(), arr_coherence.std())
            display = afmri.plot_seed_based_coherence_MNI_space(wisemean_coh, non_seed_masker_list[0], seed_coords, dirname, threshold, subject_id, cohort)
        
    #######################################
    # Build connectome in time domain     # 
    # Correlation/Covariance/Precision    #
    #                                     #
    ####################################### 

    kind_of_correlation = ['correlation', 'covariance', 'tangent', 'precision', 'partial correlation']
    print "Building connectome in Time domain : {}...\n".format(kind_of_correlation)
    # correlation and covariance return identical result
    idcor = 4
    #kind_of_analysis='time',
    #corr_matrices = afmri.build_connectome(seed_ts_subjects, kind_of_correlation=kind_of_correlation[0])
    cov_matrices = afmri.build_connectome(seed_ts_subjects, kind_of_correlation=kind_of_correlation[1])
    tangent_matrices = afmri.build_connectome(seed_ts_subjects, kind_of_correlation=kind_of_correlation[2])
    precision_matrices = afmri.build_connectome(seed_ts_subjects, kind_of_correlation=kind_of_correlation[3])
    pcorr_matrices = afmri.build_connectome(seed_ts_subjects, kind_of_correlation=kind_of_correlation[4])

    #######################################
    # Build connectome in Frequency domain# 
    # Coherence                           #
    #                                     #
    ####################################### 
    frequency_analysis = False
    if frequency_analysis is True:
      print "Building connectome in Frequency domain. Coherency...\n"
      coherency_matrices = afmri.build_connectome_in_frequency(seed_ts_subjects, pre_params, freqband)
      # convert list into ndarray subjects x time x voxels 
      coherency_matrices =  np.asarray(coherency_matrices)
      # mean across subjects
      coherency_mean_subjects = coherency_matrices.mean(axis=0)
      #######################################
      # Plot connectome in Frequency domain # 
      #######################################
      print "Plotting the mean of the coherence connectome matrices ...\n"

      msgtitle = "Mean connectome. Group:{}, Mask:{}, Corr:{} {}-{}Hz".format(cohort, mask_label.keys(),'Coherence', freqband[0], freqband[1])
      what_to_plot = OrderedDict([('plot_heatmap', True), ('plot_graph', True), ('plot_connectome',True)])
      afmri.plot_correlation_matrix(coherency_mean_subjects, label_map, msgtitle, what_to_plot)

    #######################################
    # Plot connectome in time domian      #
    # mean for subjects and/or            #
    #      single subjects                # 
    ####################################### 

    print "Plotting the mean of the connectome matrices ...\n"
    # msgtitle = "Mean connectome. Group:{}, Mask:{}, Corr:{}".format(cohort, mask_label,kind_of_correlation[idcor])
    if mask_type is 'atlas': #msdl
      msdl_atlas_dataset = datasets.fetch_atlas_msdl()
      atlas_imgs = image.iter_img(msdl_atlas_dataset.maps)
      atlas_region_coords = [plotting.find_xyz_cut_coords(img) for img in atlas_imgs]
      mcoords2plot = mcoords2plot
      mlabels = msdl_atlas_dataset.labels
      mlabels2plot = mlabels
      
      #mlabels = label_map.keys()
      #mlabels2plot = mlabels
      #mcoords2plot = label_map.values()
      msgtitle = "Mean connectome. Group:{}, Mask:{}, Corr:{}".format(cohort, mask_label,kind_of_correlation[idcor])
    else: #DMN or other  
      mlabels = mask_label.keys()
      mlabels2plot = mask_label.keys()
      mcoords2plot = mask_label.values()
      msgtitle = "Mean connectome. Group:{}, Mask:{}, Corr:{}".format(cohort, mlabels,kind_of_correlation[idcor])
    what_to_plot = OrderedDict([('plot_heatmap', True), ('plot_graph', True), ('plot_connectome',True)])
    
    if kind_of_correlation[idcor] is 'covariance':
      connectome_to_plot = cov_matrices
    elif kind_of_correlation[idcor] is 'tangent':
      connectome_to_plot = tangent_matrices
    elif kind_of_correlation[idcor] is 'precision':
      connectome_to_plot = precision_matrices
    elif kind_of_correlation[idcor] is 'partial correlation':
      connectome_to_plot = pcorr_matrices

    if type(connectome_to_plot) is list:
        connectome_to_plot = np.transpose(np.asarray(connectome_to_plot))
        connectome_to_plot_mean = connectome_to_plot.mean(-1)     
    print('Calling to afmri.plot_correlation_matrix Labels:{}',mlabels)

    afmri.plot_correlation_matrix(connectome_to_plot_mean, mlabels2plot, mcoords2plot, msgtitle, what_to_plot)
    

    # Plot connectome for individual subject
    # subject_id = 1
    # print "Plotting the connectome matrices of Group={}, Subject={}\n".format(cohort, subject_id)
    # connectome_to_plot_1s = connectome_to_plot[:,:,subject_id]
    # msgtitle = "Group:{}, Subject_{}, Mask:{}, Connectome Type:{}".format(cohort, subject_id, mask_type, kind_of_correlation[idcor])
    # afmri.plot_correlation_matrix(connectome_to_plot_1s ,mlabels2plot, mcoords2plot, msgtitle, what_to_plot)

    #######################################
    # Build Group Covariance              # 
    # and the Precision Matrix using      #
    # using GroupSparseCovarianceCV      #
    ####################################### 

    #what_to_plot = OrderedDict([('plot_heatmap', True), ('plot_graph', True), ('plot_connectome',True)])
    #
    print "Calculating the Group Covariance and the Precision Matrix (inverse covariance) \n"
    precision_matrix, cov_matrix = afmri.build_sparse_invariance_matrix(seed_ts_subjects, label_map)

    #edge_threshold = '90%'# 0.6 #'60%'
    #msgtitle = "Precision matrix:%s, edge threshold=%s" % (cohort,edge_threshold) 
    msgtitle = "Precision matrix: Cohort:%s" % (cohort) 
    print "Plotting the Precision Matrix (inverse covariance) \n"
    edge_threshold = .6
    afmri.plot_correlation_matrix(precision_matrix ,mlabels2plot, mcoords2plot, msgtitle, what_to_plot, edge_threshold)
    #afmri.plot_correlation_matrix(precision_matrix,label_map, msgtitle, what_to_plot, edge_threshold) #, edge_threshold)

    #######################################
    # Granger causality                   #
    # test and plot Granger connectome    #
    #                                     #
    ####################################### 

    print "Calculating granger causality matrix, subjects:%d Mask type:%s" %(nb_of_subjects, mask_type)
    #granger_test_results = granger_causality_analysis(seed_ts_subjects[0], pre_params,label_map, order=10)
    #Need the average , check doesnt work


    #######################################
    # Group ICA and Ward clustering       #
    #                                     #
    ####################################### 

    print('Calling to group ICA...')
    afmri.group_ICA(epi_file_list, pre_params, cohort)

    print('Calling to group Ward clustering...')
    afmri.clustering_Ward(epi_file_list, pre_params, cohort)