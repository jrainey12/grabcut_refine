import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from glob import glob
from os.path import join, basename, exists
import argparse
import os
from itertools import product


def main(base_dir,sil_dir, out_dir):
"""
Main function handles the data  inputs and output of the final silhouette.

base_dir - directory of the images.
sil_dir - directory of the silhouette images.
out_dir - directory to save the final silhouettes.
"""

	lowest_dirs = []
	for root,dirs,files in os.walk(base_dir):
   		if files and not dirs:
			lowest_dirs.append(root)
	
	lowest_sil_dirs = []
	for root,dirs,files in os.walk(sil_dir):
   		if files and not dirs:
			lowest_sil_dirs.append(root)

	lowest_dirs = sorted(lowest_dirs)
	lowest_sil_dirs = sorted(lowest_sil_dirs)
	
	for d,s in zip(lowest_dirs,lowest_sil_dirs):
			

		imgs = sorted(glob(join(d, '*.png')))
	
		masks = sorted(glob(join(s,'*.png')))
		
		#Remove any empty frames from the start and end of the image sequence.
		if not basename(imgs[0]) == basename(masks[0]):
			imgs = trim_imgs(imgs, masks[0])

		#Remove any image frames which don't have a corresponding silhouette.
		for e,(x,y) in enumerate(zip(imgs,masks)):
			
			if not basename(x) == basename(y):
				print len(imgs)
				del imgs[e]
				break
		for i,m in zip(imgs,masks):
			
			#Perform grabcut refinement.
			ref_sil = grabcut(i,m)
			print "Processed :", i, m

		
			if len(ref_sil) > 1:	
				
		
				split_base_dir = d.split('/')
				length = len(split_base_dir)

				#define the output folder.				
				out_folder = join(out_dir,split_base_dir[length-3],split_base_dir[length-2],split_base_dir[length-1])
		
				if not exists(out_folder):
						os.makedirs(out_folder)

				cv.imwrite(join(out_folder, basename(i)), ref_sil)
			else:
				print "Mask Error"

def grabcut(in_img,mask_path):
"""
Use the grabcut algorithm to refine silhouetttes.

in_img - coloured image.
mask_path - binary silhouette image to form mask.
"""
	img = cv.imread(in_img)
	
	kernel_1 = np.ones((7,7),np.uint8)

	#Create an eroded copy of the mask to form the inner foreground
	# section of the mask.
	mask_in = cv.imread(mask_path,0)
	mask_f = cv.erode(mask_in,kernel_1,iterations=1) 

	#use the input mask as the background section of the mask.
	mask_b = mask_in
	
	#New mask to save final mask to.
	mask_1 = mask_f

	height,width = mask_1.shape
	
	#Create mask for grabcut. Iterate through both masks, where both masks have
	# white pixels set to foreground, where both are black set to background and 
	# where both are different set to possible background.

	for pos in product(range(height-1), range(width-1)):
	
			pixel_f = mask_f.item(pos)
			pixel_b = mask_b.item(pos)
			
			if pixel_f == 255 and pixel_b == 255:
				mask_1[pos] = cv.GC_FGD
				
		
			elif pixel_f == 0 and pixel_b == 0:
				mask_1[pos] = cv.GC_BGD
				
			elif pixel_f == 0 and pixel_b == 255:
				mask_1[pos]  = cv.GC_PR_BGD
				
				
			
	

	bgdModel = np.zeros((1,65),np.float64)
	fgdModel = np.zeros((1,65),np.float64)
	
	#Perform grabcut operation with image and final mask.
	try:
		mask, bgdModel, fgdModel = cv.grabCut(img,mask_1,None,bgdModel,fgdModel,5,cv.GC_INIT_WITH_MASK)

	except:
		#If grabcut fails return an empty list.
		return [0]

	mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
	img = img*mask2[:,:,np.newaxis]
	
	#Convert to Grayscale and threshold image to produce the final silhouette.
	
	img_bw = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	thresh = 10
	img_thresh = cv.threshold(img_bw, thresh, 255, cv.THRESH_BINARY)[1]

	#This can be changed to "return	img" to return the coloured image rather
	# than the silhouette.
	return img_thresh



def trim_imgs(imgs, first_mask):
	
	idx = None

	first_mask_name = basename(first_mask)

	for i,x in enumerate(imgs):
		if basename(x) == first_mask_name:
			idx = i
			break	
		 
	new_imgs = imgs[idx:]
	
	return new_imgs 	
	
if __name__ == '__main__':
	
	parser = argparse.ArgumentParser(description='Refine silhouettes using grabCut. ')
	 
	parser.add_argument(
        'base_dir',
        help="Directory that contains the images to be refined.")   
    
	parser.add_argument(
        'sil_dir',
        help="Directory that contains the sils to be refined.")   
	
	parser.add_argument(
        'out_dir',
        help="Directory to save refined images to.")       
	
	args = parser.parse_args()

	main(args.base_dir, args.sil_dir,args.out_dir)
