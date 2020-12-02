# !/bin/bash

# Activate the virtualenv
source "/lrde/home2/ychen/hist_env_3.7/bin/activate"

echo "----Watershed EPM----"
parallel -j 20 ./histmapseg/build/bin/histmapseg ./data/contour_prob_tile_optimize_fix_boader.png {1} {2} ./output/test_ws/labelmap-ws-dyn_min:{1}_area_min:{2}.tif ./output/test_out/labelmap-output-dyn_min:{1}_area_min:{2}.tif ::: $(seq 0 10) ::: 50 100 200 300 400 500

# Evaluation on evaluation data
echo "----Evaluation----"
parallel -j 10 python eval_shape_detection.py ./data/gt_seg_labels.tif ./output/test_ws/labelmap-ws-dyn_min:{1}_area_min:{2}.tif -m ./data/input_mask_test.png -o ./output/PRF_test/{1}_{2} ::: $(seq 0 10) ::: 50 100 200 300 400 500

# Find the best params in evaluation
echo "----Optimum grid search----"
python grid_optimum_search.py ./output/PRF_test/

echo "----Testing score----"

python eval_shape_detection.py ./data/gt_seg_labels.tif ./output/ws/labelmap-ws-dyn_min:{7}_area_min:{400}.tif -o ./output/PRF_test/7_400

echo "----end----"
