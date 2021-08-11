from collections import defaultdict
import json
import numpy as np

from core import BoundingBox
from feature_utils import extract_box_features
from file_utils import dump_object_to_file


def convert_detection_bbox(x_min, y_min, width, height, category):
    return x_min, y_min, x_min + width - 1, y_min + height - 1, category


def convert_relationship_bbox(y_min, y_max, x_min, x_max, category):
    return x_min, y_min, x_max, y_max, category


def main():
    pair_predicate_dict = defaultdict(set)
    with open('data/detections_train.json') as file_in:
        detections_train = json.load(file_in)
    with open('data/detections_val.json') as file_in:
        detections_valid = json.load(file_in)
    with open('data/new_annotations_train.json') as file_in:
        annotations_train = json.load(file_in)
    with open('data/new_annotations_val.json') as file_in:
        annotations_valid = json.load(file_in)
    train_image_metadata = {
        item['id']: {
            'height': item['height'],
            'width': item['width']
        } for item in detections_train['images']
    }
    valid_image_metadata = {
        item['id']: {
            'height': item['height'],
            'width': item['width']
        } for item in detections_valid['images']
    }
    # train_image_metadata = dict()
    # for item in tqdm(detections_train['images']):
    #     img = cv2.imread('data/images/train_images/{}'.format(item['file_name']))
    #     height, width, _ = img.shape
    #     train_image_metadata[item['id']] = {
    #         'height': height,
    #         'width': width
    #     }
    # valid_image_metadata = dict()
    # for item in tqdm(detections_valid['images']):
    #     img = cv2.imread('data/images/valid_images/{}'.format(item['file_name']))
    #     height, width, _ = img.shape
    #     valid_image_metadata[item['id']] = {
    #         'height': height,
    #         'width': width
    #     }
    train_bbox_dict, valid_bbox_dict = defaultdict(list), defaultdict(list)
    for item in detections_train['annotations']:
        image_id, bbox, category_id = item['image_id'], item['bbox'], item['category_id']
        bbox = convert_detection_bbox(*bbox, category_id)
        assert bbox not in train_bbox_dict[image_id]
        train_bbox_dict[image_id].append(bbox)
    for item in detections_valid['annotations']:
        image_id, bbox, category_id = item['image_id'], item['bbox'], item['category_id']
        bbox = convert_detection_bbox(*bbox, category_id)
        assert bbox not in valid_bbox_dict[image_id]
        valid_bbox_dict[image_id].append(bbox)
    train_vrd_dict, valid_vrd_dict = defaultdict(list), defaultdict(list)
    for image_id in annotations_train:
        annotation_list = annotations_train[image_id]
        image_id = int(image_id.replace('.jpg', ''))
        bbox_list = train_bbox_dict[image_id]
        for annotation in annotation_list:
            subject_bbox = convert_relationship_bbox(*annotation['subject']['bbox'], annotation['subject']['category'])
            object_bbox = convert_relationship_bbox(*annotation['object']['bbox'], annotation['object']['category'])
            if subject_bbox not in bbox_list:
                bbox_list.append(subject_bbox)
                train_bbox_dict[image_id] = bbox_list
            if object_bbox not in bbox_list:
                bbox_list.append(object_bbox)
                train_bbox_dict[image_id] = bbox_list
            subject_index = bbox_list.index(subject_bbox)
            object_index = bbox_list.index(object_bbox)
            train_vrd_dict[image_id].append((subject_index, annotation['predicate'] + 1, object_index))
            pair_predicate_dict[(subject_bbox[-1], object_bbox[-1])].add(annotation['predicate'] + 1)
    for image_id in annotations_valid:
        annotation_list = annotations_valid[image_id]
        image_id = int(image_id.replace('.jpg', ''))
        bbox_list = valid_bbox_dict[image_id]
        for annotation in annotation_list:
            subject_bbox = convert_relationship_bbox(*annotation['subject']['bbox'], annotation['subject']['category'])
            object_bbox = convert_relationship_bbox(*annotation['object']['bbox'], annotation['object']['category'])
            if subject_bbox not in bbox_list:
                bbox_list.append(subject_bbox)
                valid_bbox_dict[image_id] = bbox_list
            if object_bbox not in bbox_list:
                bbox_list.append(object_bbox)
                valid_bbox_dict[image_id] = bbox_list
            subject_index = bbox_list.index(subject_bbox)
            object_index = bbox_list.index(object_bbox)
            valid_vrd_dict[image_id].append((subject_index, annotation['predicate'] + 1, object_index))
            pair_predicate_dict[(subject_bbox[-1], object_bbox[-1])].add(annotation['predicate'] + 1)
    for image_id in train_bbox_dict:
        height, width = train_image_metadata[image_id]['height'], train_image_metadata[image_id]['width']
        for i, bbox in enumerate(train_bbox_dict[image_id]):
            # train_bbox_dict[image_id][i] = (bbox[0] / width, bbox[1] / height, bbox[2] / width, bbox[3] / height, bbox[4])
            train_bbox_dict[image_id][i] = BoundingBox(
                image_id=image_id,
                x_min=bbox[0] / width,
                x_max=bbox[2] / width,
                y_min=bbox[1] / height,
                y_max=bbox[3] / height,
                category=bbox[4]
            )
    for image_id in valid_bbox_dict:
        height, width = valid_image_metadata[image_id]['height'], valid_image_metadata[image_id]['width']
        for i, bbox in enumerate(valid_bbox_dict[image_id]):
            # valid_bbox_dict[image_id][i] = (bbox[0] / width, bbox[1] / height, bbox[2] / width, bbox[3] / height, bbox[4])
            valid_bbox_dict[image_id][i] = BoundingBox(
                image_id=image_id,
                x_min=bbox[0] / width,
                x_max=bbox[2] / width,
                y_min=bbox[1] / height,
                y_max=bbox[3] / height,
                category=bbox[4]
            )
    train_data, valid_data = dict(), dict()
    for image_id in train_vrd_dict:
        bbox_list = train_bbox_dict[image_id]
        vrd_list = train_vrd_dict[image_id]
        possible_set = set((i, j) for i in range(len(bbox_list)) for j in range(len(bbox_list)) if (i != j and (bbox_list[i].category, bbox_list[j].category) in pair_predicate_dict))
        positive_set = set((item[0], item[2]) for item in vrd_list)
        negative_set = possible_set - positive_set
        train_data[image_id] = {
            'bbox_list': bbox_list,
            'box_features': np.array([extract_box_features(box) for box in bbox_list], dtype=np.float),
            'vrd_list': vrd_list,
            'negative_set': negative_set
        }
    for image_id in valid_vrd_dict:
        bbox_list = valid_bbox_dict[image_id]
        vrd_list = valid_vrd_dict[image_id]
        possible_set = set((i, j) for i in range(len(bbox_list)) for j in range(len(bbox_list)) if (i != j and (bbox_list[i].category, bbox_list[j].category) in pair_predicate_dict))
        positive_set = set((item[0], item[2]) for item in vrd_list)
        negative_set = possible_set - positive_set
        valid_data[image_id] = {
            'bbox_list': bbox_list,
            'box_features': np.array([extract_box_features(box) for box in bbox_list], dtype=np.float),
            'vrd_list': vrd_list,
            'negative_set': negative_set
        }
    dump_object_to_file(train_data, 'output/train_data.dat')
    dump_object_to_file(valid_data, 'output/valid_data.dat')
    dump_object_to_file(pair_predicate_dict, 'output/pair_predicate_dict.dat')


if __name__ == '__main__':
    main()
