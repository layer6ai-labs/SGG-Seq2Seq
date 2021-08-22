import pickle as pkl
import numpy as np
import json
import sys
from tqdm import tqdm
from joblib import Parallel, delayed
import pickle
import os

def convert(l):
    predicates_vrd = ["on", "wear", "has", "next to", "sleep next to", "sit next to", "stand next to", "park next", "walk next to", "above", "behind", "stand behind", "sit behind", "park behind", "in the front of", "under", "stand under", "sit under", "near", "walk to", "walk", "walk past", "in", "below", "beside", "walk beside", "over", "hold", "by", "beneath", "with", "on the top of", "on the left of", "on the right of", "sit on", "ride", "carry", "look", "stand on", "use", "at", "attach to", "cover", "touch", "watch", "against", "inside", "adjacent to", "across", "contain", "drive", "drive on", "taller than", "eat", "park on", "lying on", "pull", "talk", "lean on", "fly", "face", "play with", "sleep on", "outside of", "rest on", "follow", "hit", "feed", "kick", "skate on"]
    objects_vrd = ["person", "sky", "building", "truck", "bus", "table", "shirt", "chair", "car", "train", "glasses", "tree", "boat", "hat", "trees", "grass", "pants", "road", "motorcycle", "jacket", "monitor", "wheel", "umbrella", "plate", "bike", "clock", "bag", "shoe", "laptop", "desk", "cabinet", "counter", "bench", "shoes", "tower", "bottle", "helmet", "stove", "lamp", "coat", "bed", "dog", "mountain", "horse", "plane", "roof", "skateboard", "traffic light", "bush", "phone", "airplane", "sofa", "cup", "sink", "shelf", "box", "van", "hand", "shorts", "post", "jeans", "cat", "sunglasses", "bowl", "computer", "pillow", "pizza", "basket", "elephant", "kite", "sand", "keyboard", "plant", "can", "vase", "refrigerator", "cart", "skis", "pot", "surfboard", "paper", "mouse", "trash can", "cone", "camera", "ball", "bear", "giraffe", "tie", "luggage", "faucet", "hydrant", "snowboard", "oven", "engine", "watch", "face", "street", "ramp", "suitcase"]

    for i in range(len(predicates_vrd)):
        predicates_vrd[i] = predicates_vrd[i].replace(" ", "_")

    for i in range(len(objects_vrd)):
        objects_vrd[i] = objects_vrd[i].replace(" ", "_")

    predicates_vrd_dict = {}
    for i in predicates_vrd:
        i = i.replace(" ", "_")
        predicates_vrd_dict[i] = len(predicates_vrd_dict)

    object_vrd_dict = {}
    for i in objects_vrd:
        i = i.replace(" ", "_")
        object_vrd_dict[i] = len(object_vrd_dict)

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

    try:
        img_data = open("vrd_img_size_seq/" + imageID + ".pkl", "rb")
        img_data = pickle.load(img_data)
        h, w, gt = img_data
    except Exception:
        print(imageID, "has no GT!!!!!")
    #
    boxes['gt_prd_labels'] = gt['gt_prd_labels']
    boxes['gt_sbj_boxes'] = gt['gt_sbj_boxes']
    boxes['gt_sbj_labels'] = gt['gt_sbj_labels']
    boxes['gt_obj_boxes'] = gt['gt_obj_boxes']
    boxes['gt_obj_labels'] = gt['gt_obj_labels']

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
    if True:
        r = [tuple(prediction[i:i + 12]) for i in range(0, len(prediction), 12)]
        try:
            for a in r:
                b = float(a[0])
        except Exception:
            #print(r)
            print("something is wrong with the submission format")
            print(a)
            print(imageID)
            exit(0)
        r.sort()
        r = r[::-1]
        prediction = []
        for i in r:
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
        boxes['prd_scores'].append([0] * (len(predicates_vrd) + 1))
        for Confidence, RelationLabel in group_boxes[(xmins, ymins, xmaxs, ymaxs, xmino, ymino, xmaxo, ymaxo, Label1, Label2)]:
            position = predicates_vrd_dict[RelationLabel]
            boxes['prd_scores'][count][position + 1] = Confidence

        boxes['obj_scores'].append(1)
        boxes['sbj_scores'].append(1)
        boxes['sbj_labels'].append(object_vrd_dict[Label1])
        boxes['obj_labels'].append(object_vrd_dict[Label2])
        boxes['sbj_boxes'].append([xmins, ymins, xmaxs, ymaxs])
        boxes['obj_boxes'].append([xmino, ymino, xmaxo, ymaxo])
        count += 1
    for key in boxes:
        boxes[key] = np.array(boxes[key])
    boxes['image'] = id_in_pkl
    return boxes


print("Loading Submission File")
out_file = sys.argv[1]
f = open(out_file, "r")
#f.readline()  # reads off header
all_lines = [l for l in tqdm(f)]
f = open("vrd-validation-vrd.csv", "r")
f.readline()
all_ids = {}
for l in f:
    id = l.split(",")[0]
    if id not in all_ids:
        all_ids[id] = True

for l in all_lines:
    id = l.split(",")[0]
    if id in all_ids:
        all_ids[id] = False

printed = False
for id in all_ids:
    if all_ids[id]:
        if not printed:
            print("The following images don't have any predictions, scores may be inaccurate!!!!!")
            printed = True
        all_lines.append(id + ",\n")
        print(id)

final_output = Parallel(n_jobs=-1)(delayed(convert)(l) for l in tqdm(all_lines))
file = open('out.pkl', 'wb')
pickle.dump(final_output, file)
