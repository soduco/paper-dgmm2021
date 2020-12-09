# 0. Get the code
```shell script
git clone https://github.com/soduco/paper-dgmm2021.git
cd paper-dgmm2021/code/DGMM_release_code
```
You now have the code in directory paper-dgmm2021.
At this point, you should probably create a virtual environment. For instance:
```shell script
python3 -m venv dgmm-env
source ./dgmm-env/bin/activate
```
Now, install the dependencies (with pip).
```shell script
pip install -r requirements.txt
```

# 1. Get the dataset
Download the dataset from the project releases or simply by using the following shell lines.
```shell script
wget https://github.com/soduco/paper-dgmm2021/releases/download/dgmm20201-dataset/dgmm_dataset.zip
unzip dgmm_dataset.zip
```
That will download the dataset and extract it in the dgmm_dataset directory (inside the paper-dgmm2021 directory).

# 2. Create image patches for the dgmm_dataset
## Prepare the patches for training
Since the size of whole image is too big as the input of the network, we require to divide the whole map image into batches.

The batch images will save into folder *output_directory/image* and *output_directory/gt*

```shell script
python ./2.BDCN/historical_map_data/create_image_batch.py <map_image_input> <ground_truth_image_input> <output_directory>
```

If you followed the documentation to the letter, it could go like this:
```shell script
python ./2.BDCN/historical_map_data/create_image_batch.py dgmm_dataset/input/input_crop.jpg dgmm_dataset/ground_truth/lines_raster2_padded.png 2.BDCN/historical_map_data/
```

## prepare the tilling patches
Now, we will decompose the input image into tiles and put the resulting tiles in directory *2.BDCN/historical_map_data*.

```shell script
python create_tiling.py ./dgmm_dataset/input/input_crop.jpg 2.BDCN/historical_map_data
```

# 3. Decompose the dataset into train/test/val datasets
Once we get the image batches from the second step, we need to seperate the files for training, validating and testing.

The generated *.lst* files:

*train_pair.lst*: name of the files for training

*val_pair.lst*: name of the files for validating

*test_pari.lst*: name of the files for testing

```shell script
cd ./2.BDCN/historical_map_data
python prepare_training_text.py
cd ../..
```

# 4. Train

We can now train the bdcn with the prepared data. The default configuration searches for the data in the *2.BDCN/historical_map_data* directory but you can configure that (see *cfg.py*). More options are available.
```shell script
python 2.BDCN/train.py --param-dir trained-bdcn
```
The results are in the *trained-bdcn* directory.

# 5. Test

To test the bdcn on the entire image, run the following line.
```shell script
python 2.BDCN/test.py --model trained-bdcn/bdcn_100.pth --res-dir test_results
```
The resulting tiles are in the *test_results* directory.

# 6. Merge EPM patches

We just need to put everything back together with:
```shell script
python 1.Prepare_tiling/reconstruct_tiling.py ./dgmm_dataset/input/input_crop.jpg ./test_results/bdcn_100_fuse-xxx/bdcn_100_fuse/fuse/ result_epm.jpg
```
The EPM image is in *result_epm.jpg*.

# 7. Extract data within border

We should also get rid of the parts of the image outside the map area.

```shell script
python 1.Prepare_tiling/border_calibration.py result_epm.jpg ./dgmm_dataset/input/input_mask.png result_epm_mask.jpg
```
The *result_epm_mask.jpg* contains now the final EPM image. Yay!

# 8. Run watershed segmentation
Use watershed to create segmentations on the edge probability map (EPM).

## Build the watershed segmentation executable

A linux build is provided but, if for some reason it doesn't work for you, you can rebuild it yourself. Here is how.

```shell script
cd ./3.watershed/histmapseg/
mkdir newbuild && cd newbuild
conan install .. --build missing -s compiler.libcxx=libstdc++11 -s compiler.cppstd=20 -g cmake
cmake .. -DCMAKE_POSITION_INDEPENDENT_CODE=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```
Here you go: you now have your executable in *newbuild/bin/histmapseg*.

## Run the watershed segmentation executable

```shell script
./3.watershed/histmapseg/build/bin/histmapseg <input.png> <dynamic> <area_closing> <ws.tiff> <out.png>
```

*input.png*: this should be the file of EPM

*dynamic*: the parameter of dynamic

*area_closing*: the size for closing area

*ws.tiff*: the resulting watershed tiff file

*out.png*: the colorized watershed file

For instance, to use one of the parameter sets from the paper:

```shell script
./3.watershed/histmapseg/build/bin/histmapseg result_epm_mask 7 400 ws.tiff out.png
```

# 9. Evaluate the results
Evaluate the watershed segmentations with ground truth segmentations.

```shell script
python ./4.Evaluation/eval_shape_detection.py <input_gt_path> <input_contenders_path> -m <input-mask> -o <output-dir>
```

*input_gt_path*: Path to the input label map (TIFF 16 bits) for ground truth.

*input_contenders_path*: Path to the contenders label map (TIFF 16 bits) for predictions.

*input-mask*: Path to an mask image (pixel with value 0 will be discarded in the evaluation).

*output-dir*: Path to the output directory where results will be stored.

For instance, to use our example:

```shell script
python ./4.Evaluation/eval_shape_detection.py dgmm_dataset/ground_truth/gt_seg_labels.tif ws.tiff -m dgmm_dataset/input/input_mask.png -o eval/
```
