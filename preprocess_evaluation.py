from collections import defaultdict
import json
import numpy as np

from core import BoundingBox
from feature_utils import extract_box_features
from file_utils import load_object_from_file, dump_object_to_file


def main():
    with open('data/objects.json') as file_in:
        object_list = json.load(file_in)
    pair_predicate_dict = load_object_from_file('output/pair_predicate_dict.dat')
    object_dict = dict()
    for i, label in enumerate(object_list):
        object_dict['_'.join(label.split(' '))] = i
    bbox_dict = defaultdict(list)
    evaluation_data = dict()
    with open('data/detections.txt') as file_in:
        next(file_in)
        for line in file_in:
            line = line.strip().split(',')
            image_id, predictions = line[0], line[1].split(' ')
            for start_index in range(0, len(predictions), 6):
                end_index = start_index + 6
                label, confidence, x_min, y_min, x_max, y_max = predictions[start_index:end_index]
                bbox_dict[image_id].append(BoundingBox(
                    image_id=image_id,
                    x_min=float(x_min),
                    x_max=float(x_max),
                    y_min=float(y_min),
                    y_max=float(y_max),
                    category=object_dict[label],
                    confidence=float(confidence)
                ))
    for image_id in bbox_dict:
        bbox_list = bbox_dict[image_id]
        possible_set = set((i, j) for i in range(len(bbox_list)) for j in range(len(bbox_list)) if (i != j and (bbox_list[i].category, bbox_list[j].category) in pair_predicate_dict))
        evaluation_data[image_id] = {
            'bbox_list': bbox_list,
            'box_features': np.array([extract_box_features(box) for box in bbox_list], dtype=np.float),
            'vrd_list': [],
            'negative_set': possible_set
        }
    dump_object_to_file(evaluation_data, 'output/evaluation_data.dat')


if __name__ == '__main__':
    main()
