import imgaug as ia
from typing import List
import numpy as np
from label import Label, labelFromBoxes
from aug_cfg import AUG_SEQ


def augment(images: List[np.array], labels: List[Label]) -> (List[np.array], List[Label]):
    ia.seed(1)

    seq = AUG_SEQ

    labels_res = []
    images_res = []
    for label, image in zip(labels, images):
        # Augment BBs and images.
        image_aug, bbs_aug = seq(image=image, bounding_boxes=label.getBoundingBoxes())

        for i in range(len(label.getBoundingBoxes())):
            before = label.getBoundingBoxes()[i]
            after = bbs_aug.bounding_boxes[i]
            print("BB %d: (%.4f, %.4f, %.4f, %.4f) -> (%.4f, %.4f, %.4f, %.4f)" % (
                i,
                before.x1, before.y1, before.x2, before.y2,
                after.x1, after.y1, after.x2, after.y2)
                  )

        label_aug = labelFromBoxes(label.shape, label.obj_type, bbs_aug)
        labels_res.append(label_aug)
        images_res.append(image_aug)

    return images_res, labels_res
