import sys
import os
file_to_eval = sys.argv[1]
cmd =  "python multicoreconvert_seq.py " + file_to_eval
os.system(cmd)

#cmd2 = "cd /media/himanshu/himanshu-dsk2/2019openImgs/vrd/ContrastiveLosses4VRD &&"

#cmd3 = cmd2 + "python ./tools/test_net_rel.py --dataset vg --cfg configs/vg/e2e_faster_rcnn_X-101-64x4d-FPN_8_epochs_vg_v3_default_node_contrastive_loss_w_so_p_aware_margin_point2_so_weight_point5.yaml --#load_ckpt trained_models/vg_X-101-64x4d-FPN/model_step62722.pth --output_dir Outputs/vg_X-101-64x4d-FPN/ --multi-gpu-testing --do_val"

cmd3 = "python test_results.py"

os.system(cmd3)
