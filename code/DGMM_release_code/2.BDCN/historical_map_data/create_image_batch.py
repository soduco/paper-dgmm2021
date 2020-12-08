from os.path import split
from image_crop import create_save_image_batches
import argparse
import os
import warnings

dir = os.path.dirname(__file__)

def parse_args():
	parser = argparse.ArgumentParser(description='Spliting the dataset')
	parser.add_argument('image_path', type=str, default=None, 
						help='The path of input image')
	parser.add_argument('label_path', type=str, default=None, 
						help='The path of label image')
	parser.add_argument('output_path', type=str, default=None, 
						help='Output path for saving the results')
	parser.add_argument('--resize_image', type=bool, default=False, 
						help='Resize image.')
	return parser.parse_args()

if __name__ == '__main__':
	args = parse_args()
	create_save_image_batches(input_path=args.image_path, gt_path=args.label_path, output_path=args.output_path, resize_image=args.resize_image)
