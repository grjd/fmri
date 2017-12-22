#!/usr/bin/env python
# encoding: utf-8
#from __future__ import print_function
import pdb
import sys
import subprocess
import os, errno
from sys import argv
from fabric.api import run, env
import subprocess
#scp scp_transfer.py Jaime@192.168.2.68:/Users/jaime/vallecas/data/scc
def scp(source, server, path = ""):
	destination  = server + ":" + path
	commandscp = "scp -r "+ source + " "+ destination
	subprocess.call(commandscp, shell = True)
	#return not subprocess.Popen(["scp -r", source, "%s:%s" % (server, path)]).wait()
	print("scp transfer DONE!")
	return 0

def create_folder(dest_path): 
	try:
		os.makedirs(dest_path)
		print("CREATED dest_path",dest_path )
	except OSError as e:
		#pdb.set_trace()
		if e.errno != errno.EEXIST:
			raise
		if e.errno == errno.EEXIST:
			print("dest_path: ",dest_path, "already exists" )

def check_orig_ready(dest_path):
	print("Checking if file:",dest_path ,"exists")
	if os.path.exists(dest_path) is True:
		found = True 
	else:
		print("NOT Found")	
		found = False
	return found	
    	
def main(*args):
	# get list fo subjects
	import pandas as pd

	DB_PATH = "/Users/jaime/vallecas/data/scc/Database_SCD-Plus.csv"
	df = pd.read_csv(DB_PATH)
	filename_list = df['image_dir'] #["Subject_0003_17JUN1936", "Subject_0001_25NOV1940"]
	print(filename_list)
	dest_server = "jaime@192.168.2.87"
	dest_path = "/Users/jaime/Downloads/scc_image_subjects"
	orig_path = "/Volumes/Promise_Pegasus2/Vallecas/nifti"
	print "Calling to: scp ",orig_path , " ",dest_server + dest_path
	for ix, val in enumerate(filename_list):
		dest_file = os.path.join(dest_path, val)
		orig_file = os.path.join(orig_path, val)
		dest_file = dest_path
		# create folder en destination
		#create_folder(dest_file)
		#mkdir(dest_file)
		if check_orig_ready(orig_file) is True:
			print("orig file exists\n", orig_file)
			res = scp(orig_file, dest_server, dest_file)
			if res  == 0:
				print "File:", orig_file, "scp transferred successfully to",dest_server+dest_file
			else:
		  		print("File upload failed.")
		else:
			print("subject with no image, moving to the next\n",)

if __name__ == "__main__": sys.exit(main(*sys.argv[1:]))
