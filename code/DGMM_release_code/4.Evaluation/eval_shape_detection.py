#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
import logging
import cv2

import matplotlib.pyplot as plt
import numpy as np
import sys
from sklearn.metrics import auc
import evaltk

AUC_THRESHOLD_DEFAULT = 0.5
logging.basicConfig(level=logging.INFO)

BG_LABEL = 0

'''


'''
def print_scores_summary(iou_ref, iou_contender, df, AUC_threshold = 0.5, file = None):
    nlabel_gt = iou_ref.size
    nlabel_contender = iou_contender.size
    if file == None:
        file = sys.stdout

    idx = df.index.searchsorted(AUC_threshold)
    f1_auc = auc(df.index[idx:], df["F-score"].iloc[idx:])
    print(f"Number of labels in GT: {nlabel_gt}", file = file)
    print(f"Number of labels in contender: {nlabel_contender}", file = file)
    print("F1 AUC (AUC Threshold={}): {:.3f}".format(AUC_threshold, f1_auc), file=file)

    THRESHOLDS = [0.5, 0.8, 0.9, 0.95]
    idx = df.index.searchsorted(THRESHOLDS)
    subset = df.iloc[idx]
    subset.index = THRESHOLDS
    subset.index.name = "IoU"
    print(subset.round(2), file = file)

def main():
    parser = argparse.ArgumentParser(description='Evaluate the detection of shapes.')
    parser.add_argument('input_gt_path', help='Path to the input label map (TIFF 16 bits) for ground truth.')
    parser.add_argument('input_contenders_path', help='Path to the contenders label map (TIFF 16 bits) for predictions.', nargs='+')
    parser.add_argument('-m', '--input-mask', help='Path to an mask image (pixel with value 0 will be discarded in the evaluation).')
    parser.add_argument('-o', '--output-dir', help='Path to the output directory where results will be stored.')
    parser.add_argument('--auc-threshold', type=float, help='Threshold value (float) for AUC: 0.5 <= t < 1.'
                        f' Default={AUC_THRESHOLD_DEFAULT}', default=AUC_THRESHOLD_DEFAULT)

    args = parser.parse_args()

    if not (0.5 <= args.auc_threshold < 1.0):
        raise ValueError(f"auc_threshold parameter must be >= 0.5 and < 1.")

    # Load input images
    ref = cv2.imread(args.input_gt_path, cv2.IMREAD_UNCHANGED)
    if ref is None:
        raise ValueError(f"input file {args.input_gt_path} cannot be read.")

    # Load mask image
    msk_bg = None
    if args.input_mask:
        msk_bg = cv2.imread(args.input_mask, cv2.IMREAD_UNCHANGED)
        if msk_bg is None:
            raise ValueError(f"mask file {args.input_mask} cannot be read.")
        if msk_bg.shape != ref.shape:
            raise ValueError("GT and MASK image do not have the same shapes: {} vs {}", ref.shape, msk_bg.shape)
        # Create boolean mask
        msk_bg = msk_bg==0

    # Mask input image if needed
    if msk_bg is not None:
        ref = evaltk.mask_label_image(ref, msk_bg, bg_label=BG_LABEL)

    contenders = []
    for p in args.input_contenders_path:
        p = Path(p)
        contender = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if contender is None:
            raise ValueError(f"input file {p} cannot be read.")

        if contender.shape != ref.shape:
            raise ValueError("GT and PRED label maps do not have the same shapes: {} vs {}", ref.shape, contender.shape)

        # Mask predicted image if needed
        if msk_bg is not None:
            contender = evaltk.mask_label_image(contender, msk_bg, bg_label=BG_LABEL)

        contenders.append((str(p.stem), contender))


    # Create output dir early
    os.makedirs(args.output_dir, exist_ok=True)

    odir = Path(args.output_dir)
    recalls = []
    precisions = []
    for name, contender_img in contenders:
        logging.info("Processing: %s", name)
        recall, precision = evaltk.iou(ref, contender_img)
        evaltk.viz_iou(ref, recall, Path(odir, "viz_recall_{}.jpg".format(name)))
        evaltk.viz_iou(contender_img, precision, Path(odir, "viz_precision_{}.jpg".format(name)))

        df = evaltk.compute_matching_scores(recall, precision)
        df.to_csv(Path(odir, f"{name}_figure.csv"))

        with open(Path(odir, f"{name}_summary.txt"), "w") as f:
            print_scores_summary(recall, precision, df)
            print_scores_summary(recall, precision, df, file=f)

        evaltk.plot_scores(df, out = Path(odir, f"{name}_figure.pdf"))
        recalls.append(recall)
        precisions.append(precision)

    if len(contenders) == 2:
        A_recall, B_recall = recalls
        A_precision, B_precision = precisions
        (A_name, A), (B_name, B) = contenders
        A_recall_map = A_recall[ref]
        A_precision_map = A_precision[A]
        B_recall_map = B_recall[ref]
        B_precision_map = B_precision[B]
        evaltk.diff(A_recall_map, B_recall_map, out_path=Path(odir, "compare_recall.png"))
        evaltk.diff(A_precision_map, B_precision_map, out_path=Path(odir, "compare_precision.png"))


    print("All done.")


if __name__ == "__main__":
    main()
