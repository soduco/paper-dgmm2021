import os
from pathlib import Path

def train_test_val_split(image_path, gt_path, train_ratio = 0.6, val_ratio = 0.2):
	filenames = os.listdir(gt_path)
	filenames = [i.split('.')[0] for i in filenames]

	nb_image = len(filenames)
	train_list = filenames[0: int(train_ratio*nb_image)]
	val_list = filenames[int(train_ratio*nb_image): int(train_ratio*nb_image) + int(val_ratio*nb_image)]
	test_list = filenames[int(train_ratio*nb_image) + int(val_ratio*nb_image):-1]

	image_train_list = [i + '.png' for i in train_list]
	gt_train_list = [i + '.png' for i in train_list]
	
	image_val_list = [i + '.png' for i in val_list]
	gt_val_list = [i + '.png' for i in val_list]

	image_test_list = [i + '.png' for i in test_list]
	gt_test_list = [i + '.png' for i in test_list]

	data_dir = image_path[2:]
	gt_dir = gt_path[2:]

	with open(os.path.join('train_pair.lst'),'w+') as out:
		for t, g in zip(image_train_list, gt_train_list):
			out.write('{} {} \n'.format(data_dir + t, gt_dir + g))

	with open(os.path.join('val_pair.lst'),'w+') as out:
		for t, g in zip(image_val_list, gt_val_list):
			out.write('{} {} \n'.format(data_dir + t, gt_dir + g))
	
	with open(os.path.join('test_pair.lst'),'w+') as out:
		for t, g in zip(image_test_list, gt_test_list):
			out.write('{} {} \n'.format(data_dir + t, gt_dir + g))

if __name__ == '__main__':
	# Write training files
	train_dir = './image/'
	gt_dir = './gt/'
	train_test_val_split(train_dir, gt_dir)
