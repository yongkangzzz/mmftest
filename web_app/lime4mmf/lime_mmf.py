from .lime_multimodal import *
from skimage.segmentation import mark_boundaries
from skimage import img_as_ubyte
import numpy as np
from PIL import Image
import torch

#from mmf.models.mmbt import MMBT

import os
dirname = os.path.dirname(__file__)

def multi_predicts(imgs, txts, zero_image=False, zero_text=False):
    inputs = zip(imgs, txts)
    res = np.zeros((len(imgs), 2))

    for i, this_input in enumerate(inputs):

        res[i][0] = np.random.uniform(0, 1)
        res[i][1] = np.random.uniform(0, 1)

    return res

def multi_predict(imgs, txts, zero_image=False, zero_text=False):
    #model = MMBT.from_pretrained("mmbt.hateful_memes.images")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #model.to(device)
    inputs = zip(imgs, txts)
    res = np.zeros((len(imgs), 2))

    for i, this_input in enumerate(inputs):
        img = Image.fromarray(this_input[0])
        txt = this_input[1]
        #this_output = model.classify(img, txt, zero_image, zero_text)

       # res[i][this_output["label"]] = this_output["confidence"]
       # res[i][1 - this_output["label"]] = 1 - this_output["confidence"]

    return res

def lime_multimodal_explain(image_name, text):
    image_path = "static\\" + image_name
    img_try = Image.open(image_path)
    text = text
    image_numpy = np.array(img_try)
    exp1 = LimeMultimodalExplainer(image_numpy, text)
    explanation1 = exp1.explain_instance(multi_predicts, 100)
    list = explanation1.as_list()
    temp, mask = explanation1.get_image_and_mask(explanation1.top_labels[0], positive_only=True, num_features=3,
                                                 hide_rest=False)
    img_boundry = mark_boundaries(temp, mask)
    img_boundry = img_as_ubyte(img_boundry)
    print("xxx1", img_boundry)
    print("xxx2", np.uint8(img_boundry))
    PIL_image = Image.fromarray(np.uint8(img_boundry)).convert('RGB')
    exp_image = 'lime_' + image_name
    filename = os.path.join(dirname, '../static/' + exp_image)
    PIL_image.save(filename)
    print(list)
    return list, exp_image
