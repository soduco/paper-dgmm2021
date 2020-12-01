import os
from pathlib import Path

if __name__ == '__main__':
	# Write training files
	dir = str(Path(os.getcwd()))	
	train_dir = dir + '/train/data'
	gt_dir = dir + '/train/gt'
	
	train_files = os.listdir(train_dir)
	gt_files = os.listdir(gt_dir)
	
	with open(os.path.join(dir, 'train_pair.lst'),'w+') as out:
		for t, g in zip(train_files, gt_files):
			out.write('{} {}\n'.format('train/data/' + t, 'train/gt/' + g))
	
	val_dir = dir + '/val/data'
	val_gt_dir = dir + '/val/gt'
	
	val_files = os.listdir(val_dir)
	val_gt_files = os.listdir(val_gt_dir)
	
	with open(os.path.join(dir, 'val_pair.lst'),'w+') as out:
		for t, g in zip(val_files, val_gt_files):
			out.write('{} {}\n'.format('val/data/' + t, 'val/gt/' + g))

	# Write testing files
	dir = str(Path(os.getcwd()))
	test_dir = dir + '/test/data'
	
	test_files = os.listdir(test_dir)
	with open(os.path.join(dir, 'test.lst'),'w+') as out:
		for t in test_files:
			out.write('{} {}\n'.format('test/data/'+t, 'test/data/'+t))
