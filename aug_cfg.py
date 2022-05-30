import imgaug.augmenters as iaa

AUG_SEQ = iaa.Sequential([
    iaa.PerspectiveTransform(scale=(0.01, 0.15))

    #iaa.Multiply((0.7, 0.6))  # change brightness, doesn't affect BBs
    # iaa.Affine(
    #    translate_px={"x": 40, "y": 60},
    #    scale=(1, 1)
    # )  # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
])
