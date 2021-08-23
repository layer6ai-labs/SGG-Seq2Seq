import pickle as pkl
import numpy as np
import json
import sys
from tqdm import tqdm
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
import matplotlib.patches as patches
from  PIL import Image
import pickle
import os




def convert(l):
    predicates_vg = ["above", "across", "against", "along", "and", "at", "attached to", "behind", "belonging to", "between", "carrying", "covered in", "covering", "eating", "flying in", "for", "from", "growing on", "hanging from", "has", "holding", "in", "in front of", "laying on", "looking at", "lying on", "made of", "mounted on", "near", "of", "on", "on back of", "over", "painted on", "parked on", "part of", "playing", "riding", "says", "sitting on", "standing on", "to", "under", "using", "walking in", "walking on", "watching", "wearing", "wears", "with"]
    objects_vg = ["airplane", "animal", "arm", "bag", "banana", "basket", "beach", "bear", "bed", "bench", "bike", "bird", "board", "boat", "book", "boot", "bottle", "bowl", "box", "boy", "branch", "building", "bus", "cabinet", "cap", "car", "cat", "chair", "child", "clock", "coat", "counter", "cow", "cup", "curtain", "desk", "dog", "door", "drawer", "ear", "elephant", "engine", "eye", "face", "fence", "finger", "flag", "flower", "food", "fork", "fruit", "giraffe", "girl", "glass", "glove", "guy", "hair", "hand", "handle", "hat", "head", "helmet", "hill", "horse", "house", "jacket", "jean", "kid", "kite", "lady", "lamp", "laptop", "leaf", "leg", "letter", "light", "logo", "man", "men", "motorcycle", "mountain", "mouth", "neck", "nose", "number", "orange", "pant", "paper", "paw", "people", "person", "phone", "pillow", "pizza", "plane", "plant", "plate", "player", "pole", "post", "pot", "racket", "railing", "rock", "roof", "room", "screen", "seat", "sheep", "shelf", "shirt", "shoe", "short", "sidewalk", "sign", "sink", "skateboard", "ski", "skier", "sneaker", "snow", "sock", "stand", "street", "surfboard", "table", "tail", "tie", "tile", "tire", "toilet", "towel", "tower", "track", "train", "tree", "truck", "trunk", "umbrella", "vase", "vegetable", "vehicle", "wave", "wheel", "window", "windshield", "wing", "wire", "woman", "zebra"]
    predicates_vg_dict = {}
    for i in predicates_vg:
        i = i.replace(" ", "_")
        predicates_vg_dict[i] = len(predicates_vg_dict)
    
    object_vg_dict = {}
    for i in objects_vg:
        i = i.replace(" ", "_")
        object_vg_dict[i] = len(object_vg_dict)


    test_prefix = '/media/himanshu/himanshu-dsk2/2019openImgs/vrd/ContrastiveLosses4VRD/data/vg/VG_100K/'
    boxes= {}
    
    l = l.strip()
    parts = l.split(",")
    imageID = parts[0]
    id_in_pkl = imageID
    boxes = {}
    boxes['sbj_boxes'] = []
    boxes['obj_boxes'] = []
    boxes['prd_scores'] = []
    boxes['obj_scores'] = []
    boxes['sbj_scores'] = []
    boxes['sbj_labels'] = []
    boxes['obj_labels'] = []
    boxes['prd_scores'] = []
   
    img_data = open("vg_img_size_seq/" + imageID + ".pkl", "rb")
    img_data = pickle.load(img_data)
    h, w, gt = img_data

    boxes['gt_prd_labels'] = gt['gt_prd_labels']
    boxes['gt_sbj_boxes'] = gt['gt_sbj_boxes']
    boxes['gt_sbj_labels'] = gt['gt_sbj_labels']
    boxes['gt_obj_boxes'] = gt['gt_obj_boxes']
    boxes['gt_obj_labels'] = gt['gt_obj_labels']
 
    if len(parts) < 2:
        for key in boxes:
            boxes[key] = np.array(boxes[key])
        boxes['image'] = id_in_pkl
        return boxes

    prediction = parts[1].strip().split(" ")
    for i in range(0, len(prediction), 12):
        try:
            prediction[i] = float(prediction[i])
        except Exception:
            pass
    
    if len(prediction) < 2:
        for key in boxes:
            boxes[key] = np.array(boxes[key])
        boxes['image'] = id_in_pkl
        return boxes
    
    # sort the confidence
    seen_set = set([])
    if True:
        r = [tuple(prediction[i:i + 12]) for i in range(0, len(prediction), 12)]
        try:
            for a in r:
                b = float(a[0])
        except Exception:
            print(a)
            print(imageID)
            exit(0)
        r.sort()
        r = r[::-1]
        prediction = []
        cnt = 0
        non_overlap = []
        for i in r:
            if tuple(i[1:11]) not in seen_set:
                seen_set.add(tuple(i[1:11]))
                cnt += 1
            #if cnt == 101:
            #    break
            if len(i) != 12:
                print(i)
                exit(0)
            for j in i:
                prediction.append(j)
    group_boxes = {}
    for i in range(0, len(prediction), 12):
        Confidence, Label1, XMin1, YMin1, XMax1, YMax1, Label2, XMin2, YMin2, XMax2, YMax2, RelationLabel = prediction[
                                                                                                            i:i + 12]
        xmins = float(XMin1) * w
        ymins = float(YMin1) * h
        xmaxs = float(XMax1) * w
        ymaxs = float(YMax1) * h
    

        xmino = float(XMin2) * w
        ymino = float(YMin2) * h
        xmaxo = float(XMax2) * w
        ymaxo = float(YMax2) * h

        if (xmins, ymins, xmaxs, ymaxs, xmino, ymino, xmaxo, ymaxo, Label1, Label2) not in group_boxes:
            group_boxes[(xmins, ymins, xmaxs, ymaxs, xmino, ymino, xmaxo, ymaxo, Label1, Label2)] = []
        group_boxes[(xmins, ymins, xmaxs, ymaxs, xmino, ymino, xmaxo, ymaxo, Label1, Label2)].append((Confidence, RelationLabel))
        #xmin, ymin, xmax, ymax
    count = 0
    for (xmins, ymins, xmaxs, ymaxs, xmino, ymino, xmaxo, ymaxo, Label1, Label2) in group_boxes:
        boxes['prd_scores'].append([0] * (len(predicates_vg) + 1))
        for Confidence, RelationLabel in group_boxes[(xmins, ymins, xmaxs, ymaxs, xmino, ymino, xmaxo, ymaxo, Label1, Label2)]:
            position = predicates_vg_dict[RelationLabel]
            boxes['prd_scores'][count][position + 1] = Confidence

        boxes['obj_scores'].append(1)
        boxes['sbj_scores'].append(1)
        boxes['sbj_labels'].append(object_vg_dict[Label1])
        boxes['obj_labels'].append(object_vg_dict[Label2])
        boxes['sbj_boxes'].append([xmins, ymins, xmaxs, ymaxs])
        boxes['obj_boxes'].append([xmino, ymino, xmaxo, ymaxo])
        count += 1
    for key in boxes:
        boxes[key] = np.array(boxes[key])
    boxes['image'] = id_in_pkl

    return boxes


print("Loading Submission File")
#out_file = '/data2/hims44/submission_no_leak.txt'
out_file = sys.argv[1]
#f.readline()  # reads off header
#all_lines = [l for l in tqdm(f)]
f = open("vg-validation-vrd.csv", "r")
f.readline()
all_ids = {}
for l in f:
    id = l.split(",")[0]
    if id not in all_ids:
        all_ids[id] = True

f = open(out_file, "r")
f.readline()  # reads off header
all_lines = {}
for l in tqdm(f):
    id = l.split(",")[0]
    all_lines[id] = l.strip()
    if id in all_ids:
        all_ids[id] = False

printed = False
for id in all_ids:
    if all_ids[id]:
        if not printed:
            print("The following images don't have any predictions, scores may be inaccurate!!!!! Appending empty predictions.")
            printed = True
        all_lines[id] = id + ","
        print(id)

print("Done reading FIle")
f.close()

print("Changing it into pkl")

final_output = Parallel(n_jobs=-1)(delayed(convert)(all_lines[id]) for id in tqdm(all_lines))
del all_lines
print("Dumping Pickle File into out.pkl")
file = open('out.pkl', 'wb')
pickle.dump(final_output, file)
