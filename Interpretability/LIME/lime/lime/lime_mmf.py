from lime_multimodal import *
from skimage.segmentation import mark_boundaries
import numpy as np
from PIL import Image

from mmf.models.mmbt import MMBT


def multi_predict(imgs, txts, zero_image=False, zero_text=False):
    model = MMBT.from_pretrained("mmbt.hateful_memes.images")
    inputs = zip(imgs, txts)
    res = np.zeros((len(imgs), 2))

    for i, this_input in enumerate(inputs):
        img = Image.fromarray(this_input[0])
        txt = this_input[1]
        this_output = model.classify(img, txt, zero_image, zero_text)

        res[i][this_output["label"]] = this_output["confidence"]
        res[i][1 - this_output["label"]] = 1 - this_output["confidence"]

    return res

def lime_multimodal_explain(image_path,text):
    image_path = image_path
    img_try = Image.open(image_path)
    text = text
    image_numpy = np.array(img_try)
    exp1 = LimeMultimodalExplainer(image_numpy, text)
    explanation1 = exp1.explain_instance(multi_predict, 100)
    list = explanation1.as_list()
    temp, mask = explanation1.get_image_and_mask(explanation1.top_labels[0], positive_only=True, num_features=3,
                                                 hide_rest=False)
    img_boundry = mark_boundaries(temp, mask)
    return list,img_boundry
