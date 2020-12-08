import skimage
import skimage.io as skio
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.util import view_as_windows
import argparse

def generate_tiling(image_path, save_path, w_size=500):
    # Generate tiling images
    win_size = w_size
    pad_px = win_size // 2

    # Read image
    in_img = skio.imread(image_path)

    # Pad image
    img_pad = np.pad(in_img, [(pad_px,pad_px), (pad_px,pad_px), (0,0)], 'edge')

    # View as window
    tiles = view_as_windows(img_pad, (win_size,win_size,3), step=pad_px)

    for row in range(tiles.shape[0]):
        for col in range(tiles.shape[1]):
            tt = tiles[row, col, 0, ...].copy()
            # If you want black boarder, set the value to 255 (Suggest not using balck boarder)
            # bordersize=100
            # tt[:bordersize,:, 2] = 255
            # tt[-bordersize:,:, 2] = 255
            # tt[:,:bordersize, 2] = 255
            # tt[:,-bordersize:, 2] = 255
            skio.imsave(os.path.join(save_path, f"t_r{row:02d}_c{col:02d}.jpg"), tt)

def main():
    parser = argparse.ArgumentParser(description='Boarder Calibration.')
    parser.add_argument('input_path', help='Path of original image.')
    parser.add_argument('output_path', help='Directory of saving the tilings.')

    args = parser.parse_args()
    generate_tiling(args.input_path, args.output_path)

if __name__ == '__main__':
    main()
