# !/bin/bash

# Activate the virtualenv
source "/lrde/home2/ychen/hist_env_3.7/bin/activate"

echo "----EPM boarder calibration----"

python boarder_calibration.py ./data/contour_prob_tile_optimize.png ./data/input_mask_fix_epm.png ./data/contour_prob_tile_optimize_fix_boader.png

echo "----EPM2labelmap----"

parallel -j 10 python epm2labelmap.py ./data/contour_prob_tile_optimize_fix_boader.png ./data/epm_threshold/epm_label_map_{1}.png --threshold {1} ::: $(seq 0 1 10)

# Evaluation on evaluation data
echo "----shape evaluation----"
parallel -j 10 python eval_shape_detection.py ./data/gt_seg_labels.tif ./data/epm_threshold/epm_label_map_{1}.png -m ./data/input_mask_test.png -o ./output/PRF_no_watershed/{1} ::: $(seq 0 1 10)

# Find the best params in evaluation
echo "----Optimum grid search----"

python grid_optimum_search.py ./output/PRF_no_watershed/

echo "----end----"

#################################################################################

# echo "----Testing score----"

# python eval_shape_detection.py ./data/gt_seg_labels.tif ./output/ws/labelmap-ws-dyn_min:{7}_area_min:{400}.tif -m ./data/input_mask_test.png -o ./output/PRF_test/7_400
