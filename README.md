<p align="center">
<a href="https://layer6.ai/"><img src="https://github.com/layer6ai-labs/DropoutNet/blob/master/logs/logobox.jpg" width="180"></a>
</p>

## ICCV'21 Context-aware Scene Graph Generation with Seq2Seq Transformers

Authors: [Yichao Lu*](https://www.linkedin.com/in/yichaolu/), [Himanshu Rai*](https://www.linkedin.com/in/himanshu-rai-521b1318/), [Cheng Chang*](https://scholar.google.com/citations?user=X7oyRLIAAAAJ&hl=en), Boris Knyazev, [Guangwei Yu](http://www.cs.toronto.edu/~guangweiyu), Shashank Shekhar, Graham W. Taylor,  [Maksims Volkovs](http://www.cs.toronto.edu/~mvolkovs)  


## Prerequisites and Environment

* pytorch-gpu 1.13.1
* numpy 1.16.0


All experiments were conducted on a 20-core Intel(R) Xeon(R) CPU E5-2630 v4 @2.20GHz and NVIDIA V100 GPU with 32GB GPU memory.


## Dataset

### Visual Genome
Download it [here](https://drive.google.com/open?id=1VDuba95vIPVhg5DiriPtwuVA6mleYGad). Unzip it under the data folder. You should see a `vg` folder unzipped there. It contains .json annotations that suit the dataloader used in this repo.

### Visual Relation Detection

See [Images:VRD](#visual-relation-detection-1)
## Images

### Visual Genome
Create a folder for all images:
```
# ROOT=path/to/cloned/repository
cd $ROOT/data/vg
mkdir VG_100K
```
Download Visual Genome images from the [official page](https://visualgenome.org/api/v0/api_home.html). Unzip all images (part 1 and part 2) into `VG_100K/`. There should be a total of 108249 files.

### Visual Relation Detection
Create the vrd folder under `data`:
```
# ROOT=path/to/cloned/repository
cd $ROOT/data/vrd
```
Download the original annotation json files from [here](https://cs.stanford.edu/people/ranjaykrishna/vrd/) and unzip `json_dataset.zip` here. The images can be downloaded from [here](http://imagenet.stanford.edu/internal/jcjohns/scene_graphs/sg_dataset.zip). Unzip `sg_dataset.zip` to create an `sg_dataset` folder in `data/vrd`. Next run the preprocessing scripts:

```
cd $ROOT
python tools/rename_vrd_with_numbers.py
python tools/convert_vrd_anno_to_coco_format.py
```
`rename_vrd_with_numbers.py` converts all non-jpg images (some images are in png or gif) to jpg, and renames them in the {:012d}.jpg format (e.g., "000000000001.jpg"). It also creates new relationship annotations other than the original ones. This is mostly to make things easier for the dataloader. The filename mapping from the original is stored in `data/vrd/*_fname_mapping.json` where "*" is either "train" or "val".

`convert_vrd_anno_to_coco_format.py` creates object detection annotations from the new annotations generated above, which are required by the dataloader during training.

## Pre-trained Object Detection Models
Download pre-trained object detection models [here](https://drive.google.com/open?id=1NrqOLbMa_RwHbG3KIXJFWLnlND2kiIpj). Unzip it under the root directory. **Note:** We do not include code for training object detectors. Please refer to the "(Optional) Training Object Detection Models" section in [Large-Scale-VRD.pytorch](https://github.com/jz462/Large-Scale-VRD.pytorch) for this.

## Our Trained Relationship Detection Models
Download our trained models [here](https://drive.google.com/open?id=15w0q3Nuye2ieu_aUNdTS_FNvoVzM4RMF). Unzip it under the root folder and you should see a `trained_models` folder there.

## Directory Structure
The final directories should look like:
```
|-- data
|   |-- detections_train.json
|   |-- detections_val.json
|   |-- new_annotations_train.json
|   |-- new_annotations_val.json
|   |-- objects.json
|   |-- predicates.json
|-- evaluation
|-- output
|   |-- pair_predicate_dict.dat
|   |-- train_data.dat
|   |-- valid_data.dat
|-- config.py
|-- core.py
|-- data_utils.py
|-- evaluation_utils.py
|-- feature_utils.py
|-- file_utils.py
|-- preprocess.py
|-- trainer.py
|-- transformer.py
```

## Evaluating Pre-trained Relationship Detection models

DO NOT CHANGE anything in the provided config files(configs/xx/xxxx.yaml) even if you want to test with less or more than 8 GPUs. Use the environment variable `CUDA_VISIBLE_DEVICES` to control how many and which GPUs to use. Remove the
`--multi-gpu-test` for single-gpu inference.


### Visual Genome
**NOTE:** May require at least 64GB RAM to evaluate on the Visual Genome test set

We use three evaluation metrics for Visual Genome:
1. SGDET: predict all the three labels and two boxes
1. SGCLS: predict subject, object and predicate labels given ground truth subject and object boxes
1. PRDCLS: predict predicate labels given ground truth subject and object boxes and labels

To test a trained model using a VGG16 backbone with "SGDET", run
```
python ./tools/test_net_rel.py --dataset vg --cfg configs/vg/e2e_faster_rcnn_VGG16_8_epochs_vg_v3_default_node_contrastive_loss_w_so_p_aware_margin_point2_so_weight_point5_no_spt.yaml --load_ckpt trained_models/vg_VGG16/model_step62722.pth --output_dir Outputs/vg_VGG16 --multi-gpu-testing --do_val
```
Use `--use_gt_boxes` option to test it with "SGCLS"; use `--use_gt_boxes --use_gt_labels` options to test it with "PRDCLS". The results will vary slightly with the last line of Table 6 in the paper.

To test a trained model using a vg_X-101-64x4d-FPN backbone with "SGDET", run
```
python ./tools/test_net_rel.py --dataset vg --cfg configs/vg/e2e_faster_rcnn_X-101-64x4d-FPN_8_epochs_vg_v3_default_node_contrastive_loss_w_so_p_aware_margin_point2_so_weight_point5.yaml --load_ckpt trained_models/vg_X-101-64x4d-FPN/model_step62722.pth --output_dir Outputs/vg_X-101-64x4d-FPN --multi-gpu-testing --do_val
```
Use `--use_gt_boxes` option to test it with "SGCLS"; use `--use_gt_boxes --use_gt_labels` options to test it with "PRDCLS". The results will vary slightly with those at the last line of Table 1 in the supplementary.

### Visual Relation Detection
To test a trained model initialized by an ImageNet pre-trained VGG16 model, run
```
python ./tools/test_net_rel.py --dataset vrd --cfg configs/vrd/e2e_faster_rcnn_VGG16_16_epochs_vrd_v3_default_node_contrastive_loss_w_so_p_aware_margin_point2_so_weight_point5_IN_pretrained.yaml --load_ckpt trained_models/vrd_VGG16_IN_pretrained/model_step7559.pth --output_dir Outputs/vrd_VGG16_IN_pretrained --multi-gpu-testing --do_val
```
The results are slightly different with those at the second to the last line of Table 7.

To test a trained model initialized by an COCO pre-trained VGG16 model, run
```
python ./tools/test_net_rel.py --dataset vrd --cfg configs/vrd/e2e_faster_rcnn_VGG16_16_epochs_vrd_v3_default_node_contrastive_loss_w_so_p_aware_margin_point2_so_weight_point5_COCO_pretrained.yaml --load_ckpt trained_models/vrd_VGG16_COCO_pretrained/model_step7559.pth --output_dir Outputs/vrd_VGG16_COCO_pretrained --multi-gpu-testing --do_val
```
The results are slightly different with those at the last line of Table 7.

## Training Scene Graph Generation Models

With the following command lines, the training results (models and logs) should be in `$ROOT/Outputs/xxx/` where `xxx` is the .yaml file name used in the command without the ".yaml" extension. If you want to test with your trained models, simply run the test commands described above by setting `--load_ckpt` as the path of your trained models.

### Visual Relation Detection
To train our scene graph generation model on the VRD dataset, run
```
python preprocess.py

python trainer.py --num-encoder-layers 4 --num-decoder-layers 2 --nhead 4 --num-epochs 500 --learning-rate 1e-3

python preprocess_evaluation.py

python write_prediction.py
```

### Visual Genome
To train our scene graph generation model on the VG dataset, download the json files from https://visualgenome.org/api/v0/api_home.html, put the extracted files under `data` and then run
```
python preprocess.py

python trainer.py --num-encoder-layers 4 --num-decoder-layers 2 --nhead 4 --num-epochs 2000 --learning-rate 1e-3

python preprocess_evaluation.py

python write_prediction.py
```

## Acknowledgements
This repository uses code based on the [Neural-Motifs](https://github.com/rowanz/neural-motifs) source code from Rowan Zellers, as well as
code from the [Detectron.pytorch](https://github.com/roytseng-tw/Detectron.pytorch) repository by Roy Tseng. See LICENSES for additional details.


## Citation

If you find this code useful in your research, please cite the following paper:

    @inproceedings{lu2021seq2seq,
      title={Context-aware Scene Graph Generation with Seq2Seq Transformers},
      author={Yichao Lu, Himanshu Rai, Jason Chang, Boris Knyazev, Shashank Shekhar, Graham W. Taylor, Maksims Volkovs},
      booktitle={ICCV},
      year={2021}
    }

 
