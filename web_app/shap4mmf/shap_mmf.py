from mmf.models.mmbt import MMBT
import torch
from shap4mmf._explainer import Explainer
import numpy as np
from PIL import Image

import os
dirname = os.path.dirname(__file__)

def predict_HM(image_name, text):
    model = MMBT.from_pretrained("mmbt.hateful_memes.images")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    image = "static\\" + image_name
    output = model.classify(image, text)
    # Explainer hyper params
    max_evals = 100
    batch_size = 50
    explainer = Explainer(model, max_evals=max_evals, batch_size=batch_size)

    target_images = Image.open(image)
    target_images = np.array(target_images, dtype=np.uint8)
    if target_images.shape[2] > 3:
        target_images = target_images[:, :, :3]
    target_images = target_images.reshape(1, target_images.shape[0], target_images.shape[1], target_images.shape[2])
    target_texts = np.array([text])
    image_shap_values = explainer.explain(target_images, target_texts, "image_only")
    PIL_image = explainer.image_plot(image_shap_values)
    exp_image = 'shap_' + image_name
    filename = os.path.join(dirname, '../static/' + exp_image)
    PIL_image.save(filename)
    # hateful = "hateful" if output["label"] == 1 else "not hateful"
    # result = "This image is: " + hateful + ". " + f"Model's confidence: {output['confidence'] * 100:.3f}%"
    result = []
    hateful = "Hateful" if output["label"] == 1 else "Not Hateful"
    result.append("This image is: " + hateful)
    result.append(f"Model's confidence: {output['confidence'] * 100:.3f}%")
    return result, exp_image