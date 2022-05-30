from typing import Tuple, List

import imgaug as ia
from imgaug.augmentables.bbs import BoundingBoxesOnImage


class CWHBoundingBox(ia.BoundingBox):
    def __init__(self, center_x, center_y, width, height, label=None):
        super().__init__(
            x1=center_x - width / 2,
            y1=center_y + height / 2,
            x2=center_x + width / 2,
            y2=center_y - height / 2,
            label=label
        )


class Label:
    def __init__(self, shape: Tuple[int, int], obj_type: List[str], x: List[float], y: List[float], width: List[float],
                 height: List[float]):
        self.obj_type = obj_type
        self.shape = shape
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def getBoundingBoxes(self) -> [CWHBoundingBox]:
        boxes = zip(self.x, self.y, self.width, self.height)
        return BoundingBoxesOnImage(
            [CWHBoundingBox(x, y, width, height) for x, y, width, height in boxes], shape=self.shape)

    def toStr(self):
        objects = zip(self.obj_type, self.x, self.y, self.width, self.height)
        return [f"{int(t)} {x} {y} {width} {height}\n" for t, x, y, width, height in objects]


def labelFromBoxes(shape, classes, boxes: BoundingBoxesOnImage) -> Label:
    type_vals = []
    x_vals = []
    y_vals = []
    width_vals = []
    height_vals = []
    i = 0
    for box in boxes.bounding_boxes:
        obj_type, x, y, width, height = classes[i], box.center_x, box.center_y, box.width, box.height
        type_vals.append(obj_type)
        x_vals.append(x)
        y_vals.append(y)
        width_vals.append(width)
        height_vals.append(height)
        i += 1

    return Label(shape, type_vals, x_vals, y_vals, width_vals, height_vals)
