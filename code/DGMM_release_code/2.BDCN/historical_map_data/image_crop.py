# import the necessary packages
import numpy as np
import os

import cv2
from PIL import Image

def create_save_image_batches(input_path, gt_path, output_path, resize_image=False, image_size_x=500, image_size_y=500):
	'''
	Description: Crop and save image batches into files
	Parameters
	----------
	input_path: image path
	output_path: output path
	image_size_x, image_size_y: the size of height and width of the images

	Returns
	-------
	None
	'''

	# Read image check directory exist or not
	dir_save_image_batch = os.path.join(output_path + 'image')
	dir_save_gt_batch = os.path.join(output_path + 'gt')

	if not os.path.exists(dir_save_image_batch):
		os.makedirs(dir_save_image_batch)
		print("Directory ", dir_save_image_batch, " Created ")
	else:
		print("Directory ", dir_save_image_batch, " already exists")
	
	if not os.path.exists(dir_save_gt_batch):
		os.makedirs(dir_save_gt_batch)
		print("Directory ", dir_save_gt_batch, " Created ")
	else:
		print("Directory ", dir_save_gt_batch, " already exists")

	######################################################################################
	
	if resize_image:
		# Resize original image
		file_original_image =  np.array(cv2.imread(input_path))
		image_shape = file_original_image.shape
		image_resize_shape_x = int(np.ceil(image_shape[0] / image_size_x)) * image_size_x
		image_resize_shape_y = int(np.ceil(image_shape[1] / image_size_y)) * image_size_y
		new_image = np.zeros((image_resize_shape_x, image_resize_shape_y, 3)).astype('uint8')
		new_image[0:image_shape[0], 0:image_shape[1], :] = file_original_image
		save_original_image_path = '.' + input_path.split('.')[1] + '_crop_image.png'
		cv2.imwrite(save_original_image_path, new_image)
		print("Save resize image into ", save_original_image_path)

		# Resize gt image
		gt_image = np.array(cv2.imread(gt_path))
		image_shape = gt_image.shape
		image_resize_shape_x = int(np.ceil(image_shape[0] / image_size_x)) * image_size_x
		image_resize_shape_y = int(np.ceil(image_shape[1] / image_size_y)) * image_size_y
		new_image = np.zeros((image_resize_shape_x, image_resize_shape_y, 3)).astype('uint8')
		new_image[0:image_shape[0], 0:image_shape[1], :] = gt_image
		save_gt_path ='.' + gt_path.split('.')[1] + '_crop_gt.png'
		cv2.imwrite(save_gt_path, new_image)
		print("Save resize gt into ", save_gt_path)
	else:
		pass

	# create image batches
	crop(input_path, dir_save_image_batch, image_size_x, image_size_y)
	# Create gt batches
	crop(gt_path, dir_save_gt_batch, image_size_x, image_size_y)


def crop(input, output_path, height, width):
	k = 0
	im = Image.open(input)
	imgwidth, imgheight = im.size
	for i in range(0, imgheight, height):
		for j in range(0, imgwidth, width):
			box_str = str(j) + '_' + str(i) + '_' + str(j+width) + '_' + str(i+height)
			box = (j, i, j+width, i+height)
			a = im.crop(box)
			a.save(os.path.join(output_path, "%s_%s.png" % (k, box_str)))
			k +=1
