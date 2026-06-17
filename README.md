# Heatmap analysis of CNNs used to predict referable diabetic retinopathy

This repository contains all the code to create the models and run analysis on the results.

## Datasets
Data in ths study are sourced from:
* DDR Dataset: https://github.com/nkicsl/DDR-dataset
* BRSET Dataset: https://github.com/luisnakayama/BRSET
* Kaggle Resized 2015 & 2019 blindness detection images dataset: https://www.kaggle.com/datasets/benjaminwarner/resized-2015-2019-blindness-detection-images

Some images in the DDR are graded incorrectly so have been adjusted as shown below, with reasoning in brackets.

Also: 20170622080837660.jpg severe --> PDR (NVE)
* 007-2764-100.jpg mild --> moderate (exudates)
* 20170519153500194.jpg mild --> moderate (haemorrhages)
* 20170518171333730.jpg mild --> moderate (haemorrhages)
* 007-3036-100.jpg mild --> moderate (haemorrhages)
* 007-2976-100.jpg mild --> moderate (haemorrhages)
* 007-2852-100.jpg mild --> moderate (haemorrhages)
* 007-2836-100.jpg mild --> moderate (haemorrhages)
* 007-2787-100.jpg mild --> moderate (haemorrhages)
* 007-2769-100.jpg mild --> moderate (haemorrhages)
* 007-2684-100.jpg mild --> moderate (haemorrhages)
* 007-2682-100.jpg mild --> moderate (haemorrhages)
* 007-5728-300.jpg moderate --> severe (IRMA)

### Directory structure
There are two model types: `grade` and `refer`.
Each type needs its own folder (`grade` or `refer`) containing a numerical directory for each grade.
For `refer`, this will be `0` (non-referable) and `1` (referable).
For `grade`, this is from `0` (no DR) to `4` (proliferative DR).
Place images from the datasets above into their corresponding folders.
It is recommended that grades within each model type have the same number of images.

To make training quicker, we can pre-resize images to prevent the model having to resize at each iteration.
Use the script `util/create_resized.sh <refer|grade>` to resize these images, which will be stored in a new folder (e.g. `refer_224`).
During training and some analysis routines, these folders are copied to ramdisk to improve processing times.

## Annotations
The heatmap set uses annotated images from the DDR dataset mentioned above, with additional annotations from Murphy et al. (2024): https://doi.org/10.3390/jcm13030807

## Setup
1. Create a ramdisk at `/mnt/ramdisk`, or create folder at this path to make it
   valid. The code expects this path to exist.
2. Required python packages are included in the Docker container by default.
   Running `./docker_bash.sh` will open a bash session with all of the required
   output files and scripts accessible, and outputs will be accessible outside
   of the Docker session. Alternatively, install the python packages in your
   usual environment by running `pip install -r src/requrements.txt` and run
   scripts as usual. Both options will give the same outputs.

## Refer models
1. Edit `src/create_heatmap.py` line 69 to be `LESION_LABELS = LESION_LABELS_REFER`.
2. `./train_docker.sh refer "" --base_only`
3. `util/test_all_models.sh trained_refer accuracy_refer refer`
4. `src/generate_all_heatmaps.py --model_csv model_stats_refer.csv --image_dir heatmap_images --output_dir heatmaps_refer --cmap viridis`
5. `util/dice_everything.sh refer`
6. `src/heatmap_overlap_graph.py --dice_stats_csv dice_lesions_refer/*/*otsu.csv --output_png dice_graph_refer.png --title "Proportion of DR features covered by refer model heatmaps"`

## Grade models
1. Edit `src/create_heatmap.py` line 69 to be `LESION_LABELS = LESION_LABELS_GRADE`.
2. `./train_docker.sh grade "" --base_only`
3. Edit `src/create_heatmap.py` line 69 to be `LESION_LABELS = LESION_LABELS_REFER`.
4. `util/test_all_models.sh trained_grade accuracy_grade grade`
5. `util/grade_to_refer.sh accuracy_grade/*.csv`
6. `src/grade_to_refer_stats.py`
7. `src/generate_all_heatmaps.py --model_csv model_stats_grade_to_refer.csv --image_dir heatmap_images --output_dir heatmaps_grade_to_refer --grade_to_refer --cmap viridis`
8. `util/dice_everything.sh grade`
9. `src/heatmap_overlap_graph.py --dice_stats_csv dice_lesions_grade/*/*otsu.csv --output_png dice_graph_grade.png --title "Proportion of DR features covered by grade model heatmaps"`

## Additional heatmaps
```
python src/csv_to_heatmap.py --heatmap_csv heatmaps_refer/*1.csv --cmap viridis --output_dir heatmaps_refer/heatmaps_nolabel --combined_output_png combined_refer_nolabel.png
python src/csv_to_heatmap.py --heatmap_csv heatmaps_refer/*1.csv --cmap viridis --output_dir heatmaps_refer/heatmaps --combined_output_png combined_refer.png --infer_label
python src/csv_to_heatmap.py --heatmap_csv heatmaps_grade_to_refer/*1.csv --cmap viridis --output_dir heatmaps_grade_to_refer/heatmaps_nolabel --combined_output_png combined_grade_to_refer_nolabel.png
python src/csv_to_heatmap.py --heatmap_csv heatmaps_grade_to_refer/*1.csv --cmap viridis --output_dir heatmaps_grade_to_refer/heatmaps --combined_output_png combined_grade_to_refer.png --infer_label
```

## Additional images
```
python src/plot_dice_for_lesion.py --dice_stats_csv dice_lesions_refer/grade_4/dice_stats_otsu.csv --lesion NVD --output_png dice_refer_nvd.png --title "Proportion of Neovascularisation at the Disc Covered by Refer Heatmaps" --boxplot
python src/plot_dice_for_lesion.py --dice_stats_csv dice_lesions_grade/grade_4/dice_stats_otsu.csv --lesion NVD --output_png dice_grade_nvd.png --title "Proportion of Neovascularisation at the Disc Covered by Grade Heatmaps" --boxplot

```

## AUROC values
```
python src/model_roc.py --accuracy_csv accuracy_refer/*.csv
python src/model_roc.py --accuracy_csv accuracy_grade_to_refer/*.csv
```
