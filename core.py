from typing import Union


class BoundingBox(object):
    __slots__ = [
        'image_id',
        'x_min',
        'y_min',
        'x_max',
        'y_max',
        'category',
        'confidence',
        'x_center',
        'y_center'
    ]

    def __init__(
            self,
            image_id: Union[int, str],
            x_min: float,
            x_max: float,
            y_min: float,
            y_max: float,
            category: int,
            confidence: float = 1.0
    ):
        if not (0.0 <= x_min <= x_max <= 1.0 and 0.0 <= y_min <= y_max <= 1.0):
            x_min = max(min(x_min, 1.0), 0.0)
            y_min = max(min(y_min, 1.0), 0.0)
            x_max = max(min(x_max, 1.0), 0.0)
            y_max = max(min(y_max, 1.0), 0.0)
        self.image_id = image_id
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.category = category
        self.confidence = confidence
        self.x_center = (x_min + x_max) / 2
        self.y_center = (y_min + y_max) / 2

    def get_area(self):
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)


class VisualRelationship(object):
    __slots__ = [
        'image_id',
        'index_1',
        'index_2',
        'box_1',
        'box_2',
        'category'
    ]

    def __init__(
            self,
            image_id: int,
            index_1: int,
            index_2: int,
            box_1: BoundingBox,
            box_2: BoundingBox,
            category: int
    ):
        assert image_id == box_1.image_id == box_2.image_id
        self.image_id = image_id
        self.index_1 = index_1
        self.index_2 = index_2
        self.box_1 = box_1
        self.box_2 = box_2
        self.category = category
