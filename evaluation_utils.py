def evaluate_per_image_recall(data, prediction_dict):
    recall_list = [0] * 5  # 5, 10, 20, 50, 100
    count = 0
    for image_id in prediction_dict:
        predictions = [item[:3] for item in prediction_dict[image_id]]
        targets = set(data[image_id]['vrd_list'])
        if len(targets) == 0:
            continue
        for i, k in enumerate([5, 10, 20, 50, 100]):
            recall_list[i] += len(set(predictions[:k]) & targets) / len(targets)
        count += 1
    for i in range(len(recall_list)):
        recall_list[i] /= count
    return recall_list
