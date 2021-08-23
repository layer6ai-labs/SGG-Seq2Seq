import json

from file_utils import load_object_from_file


def main():
    with open('data/objects.json') as file_in:
        object_list = json.load(file_in)
    with open('data/predicates.json') as file_in:
        relation_list = json.load(file_in)
    object_list = ['_'.join(item.split(' ')) for item in object_list]
    vrd_data = load_object_from_file('output/evaluation_data.dat')
    prediction_dict = load_object_from_file('cache/evaluation_prediction_dict.dat')
    file_out = open('prediction.txt', 'w+')
    for image_id in prediction_dict:
        result_list = []
        bbox_list = vrd_data[image_id]['bbox_list']
        for subject_index, relation_id, object_index, confidence_score in prediction_dict[image_id]:
            subject_bbox = bbox_list[subject_index]
            relation_label = '_'.join(relation_list[relation_id - 1].split(' '))
            object_bbox = bbox_list[object_index]
            result_list.append(f'{confidence_score} {object_list[subject_bbox.category]} {subject_bbox.x_min} {subject_bbox.y_min} {subject_bbox.x_max} {subject_bbox.y_max} {object_list[object_bbox.category]} {object_bbox.x_min} {object_bbox.y_min} {object_bbox.x_max} {object_bbox.y_max} {relation_label}')
        file_out.write('{},{}\n'.format(image_id, ' '.join(result_list)))


if __name__ == '__main__':
    main()
