import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from glob import glob
from os.path import join, basename, exists
import argparse
import os

np.set_printoptions(threshold=np.nan)

def main(base_dir,sil_dir, out_dir):
	

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
	#print lowest_dirs
	#print len(lowest_dirs)
	for d,s in zip(lowest_dirs,lowest_sil_dirs):
			

		imgs = sorted(glob(join(d, '*.png')))
	
		masks = sorted(glob(join(s,'*.png')))
		
		if not basename(imgs[0]) == basename(masks[0]):
			imgs = trim_imgs(imgs, masks[0])


		for e,(x,y) in enumerate(zip(imgs,masks)):
			print e
			if not basename(x) == basename(y):
				print len(imgs)
				del imgs[e]
				break
		for i,m in zip(imgs,masks):

			ref_sil = grabcut(i,m)
			print "Processed :", i, m

			if not ref_sil == None:
					
				split_base_dir = d.split('/')
				length = len(split_base_dir)
				#print split_base_dir
				out_folder = join(out_dir,split_base_dir[length-3],split_base_dir[length-2],split_base_dir[length-1])
		
				if not exists(out_folder):
						os.makedirs(out_folder)

				cv.imwrite(join(out_folder, basename(i)), ref_sil)
			else:
				print "Mask Error"

def grabcut(in_img,mask_path):
	
	img = cv.imread(in_img)
	#img = cv.imread(mask_path)
	kernel_1 = np.ones((5,5),np.uint8)
	kernel_2 = np.ones((10,10),np.uint8)
	mask_in = cv.imread(mask_path,0)
	mask_f = cv.erode(mask_in,kernel_1,iterations=1) 
	mask_b = cv.dilate(mask_in,kernel_2,iterations=1)


	mask_1 = mask_f
	height,width = mask_1.shape
	for x in range(0,(width-1)):
		for y in range(0,height-1):
			if mask_f[y,x] == 255 and mask_b[y,x] == 255:
				mask_1[y,x] = cv.GC_FGD
				#mask_1[y,x] = 255
		
			elif mask_f[y,x] == 0 and mask_b[y,x] == 0:
				mask_1[y,x] = cv.GC_BGD
				#mask_1[y,x] = 0
			elif mask_f[y,x] == 0 and mask_b[y,x] == 255:
				mask_1[y,x] = cv.GC_PR_BGD
				#mask_1[y,x] = 127

	#print mask_1
#	plt.imshow(mask_1)
#	plt.show()

	#return None
	#height,width = mask_f.shape
	#for x in range(0,(width-1)):
	#	for y in range(0,height-1):
	#		#print mask[y,x]
	#		if mask_f[y,x] == 255:
	#			mask_f[y,x] = cv.GC_FGD
	#		else: 
	#			mask_f[y,x] = cv.GC_BGD

	bgdModel = np.zeros((1,65),np.float64)
	fgdModel = np.zeros((1,65),np.float64)
	#rect = (175,45,80,160)
	#mask[mask == 0] = 0
	#mask[mask == 255] = 1
	#try:
	mask, bgdModel, fgdModel = cv.grabCut(img,mask_1,None,bgdModel,fgdModel,5,cv.GC_INIT_WITH_MASK)

	#except:
	#	#"Mask Error"
	#	return None
	mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
	img = img*mask2[:,:,np.newaxis]
	
	img_bw = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	thresh = 10
	img_thresh = cv.threshold(img_bw, thresh, 255, cv.THRESH_BINARY)[1]

	#plt.imshow(img_thresh)
	#plt.show()

#	plt.imshow(img)
#	plt.show()

	return img_thresh

def old_grabcut(sil,mask_path):
	
	img = cv.imread(sil)
	kernel = np.ones((3,3),np.uint8)
	mask = cv.imread(mask_path,0)
	mask = cv.erode(mask,kernel,iterations=1) 
	height,width = mask.shape
	for x in range(0,(width-1)):
		for y in range(0,height-1):
			#print mask[y,x]
			if mask[y,x] == 255:
				mask[y,x] = cv.GC_FGD
				#print mask[y,x]
			else: 
				mask[y,x] = cv.GC_BGD
	bgdModel = np.zeros((1,65),np.float64)
	fgdModel = np.zeros((1,65),np.float64)
	#rect = (175,45,80,160)
	#mask[mask == 0] = 0
	#mask[mask == 255] = 1
	try:
		mask, bgdModel, fgdModel = cv.grabCut(img,mask,None,bgdModel,fgdModel,5,cv.GC_INIT_WITH_MASK)

	except:
		#"Mask Error"
		return None
	mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
	img = img*mask2[:,:,np.newaxis]
	#plt.imshow(img)
	#plt.show()
	return img


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
	#joint_generator(args.base_dir)
	main(args.base_dir, args.sil_dir,args.out_dir)
