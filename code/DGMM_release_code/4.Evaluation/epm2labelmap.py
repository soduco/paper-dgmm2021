#!/usr/bin/env python3

import argparse
import numpy as np
import cv2

def main():
    parser = argparse.ArgumentParser(description='Takes an edge probability map and produces a label map. Edge pixels > 0.')
    parser.add_argument('input_path', help='Path to the input EPM (PNG format or TIFF 16 bits).')
    parser.add_argument('output_path', help='Path to the output label map (TIFF 16 format).')
    parser.add_argument('--threshold', type=float, help='Threshold value (float): v<=threshold => v in background.', default=0.0)
    parser.add_argument('--debug_labels', help='Path to debug image (JPG) where to save the RGB label map.')

    args = parser.parse_args()

    if args.threshold < 0.0:
        raise ValueError(f"threshold parameter must be > 0.")

    # Load input image
    in_img = cv2.imread(args.input_path, cv2.IMREAD_UNCHANGED)
    if in_img is None:
        raise ValueError(f"input file {args.input_path} cannot be read.")
    if in_img.ndim > 2:
        in_img = in_img[...,0]
    
    # Compute binary image
    bin_img = in_img <= args.threshold

    # Extract connected components
    retval, labels = cv2.connectedComponents(bin_img.astype(np.uint8), connectivity=4) # , ltype=cv2.CV_16U
    print(f"Processed {args.input_path}. Found {retval} components.")
    print(f"max(labels)={np.max(labels)}, labels.dtype={labels.dtype}")

    # Save the output
    cv2.imwrite(args.output_path, labels.astype(np.uint16))

    # Extra, optional debub output
    if args.debug_labels is not None:
        import matplotlib.cm
        nelem = retval
        cmap = matplotlib.cm.get_cmap(name='hsv', lut=nelem)
        np.random.seed(0)
        perms = np.random.permutation(nelem)
        lut_rgba = cmap(perms, bytes=True)
        lut_rgb = lut_rgba[...,:3]
        lut_rgb[0] = (0,0,0)
        lut_rgb[1] = (128,128,128)
        debug_img = lut_rgb[labels]
        cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR, dst=debug_img)
        cv2.imwrite(args.debug_labels, debug_img)

if __name__ == "__main__":
    main()
