"""
===============================================================================
shap_explainer.py

explainer class for singlemodal or multimodal inputs
===============================================================================
"""
import shap
import numpy as np
import torch
import copy
import shap4mmf._utils as utils
from shap4mmf._text_plot import text as modified_text_plot

from mmf.models.mmbt import MMBT
from transformers import AutoTokenizer

from PIL import Image
from typing import List

from ._plot import image


class Explainer(object):
    """Use shap as the explaination engine for singlemodal and multimodal models

    Attributes:
    """

    def __init__(self, model: "MMF Model", algorithm="partition",
                 max_evals=None, batch_size=None
                 ):
        self._supported_algos = ("partition", )

        self._supported_modes = ("text_only", "image_only", )
        # input validations

        if algorithm not in self._supported_algos:
            raise ValueError(f"This algotithm {algorithm} is not supported!")

        # model should have a .classify method
        if not hasattr(model, 'classify'):
            raise ValueError(f"Model object must have a .classify attribute.")

        # public features
        self.model = model
        self.algorithm = algorithm
        # some methods may allow speeding up
        self.max_evals = max_evals if max_evals is not None else 100
        self.batch_size = batch_size if batch_size is not None else 50

        # internal features
        self._fixed_images = None
        self._fixed_texts = None

#    def _f_multimodal(self, images, texts):
#        """ TODO (WIP): Multimodal funtion for shap to explain
#
#        Args:
#            param1: 
#
#        Returns:
#            what is returned: 
#
#        """
#        pass

    def _f_image(self, images: np.ndarray):
        """ Image-only function for shap to explain

        Args:
            images: np.ndarray of shape (N, D1, D2, C);
                N = number of samples
                D1, D2, C = three channel image

        Returns:
            numpy array of predictions with shape = (N, 2);
                - N[i] = score of the image being i
        """
        out = np.zeros((len(images), 2))  # output same shape

        # inputs neeeds to be [PIL.Image.Image]; if not try to transform
        if not isinstance(images[0], Image.Image):
            images = utils.arr_to_img(images)

        for i, (image, text) in enumerate(zip(images, self._fixed_texts)):
            # classify, output is a tupe (index, score)
            ind, score = self.model.classify(image, text).values()
            out[i][ind] = score
            out[i][1 - ind] = 1 - score

        return out
        # test if only output is the probability of being hateful
        # return out[:, 1][:, np.newaxis]

    def _f_text(self, texts: np.ndarray):
        """ Text-only function for shap to explain
        testing only texts
        Args:
            texts: np.ndarray of strings, shape = (N,)
                N = number of samples

        Returns:
            numpy array of predictions with shape = (N, 2);
                - N[i] = score of the image being i
        """
        out = np.zeros((len(texts), 2))  # output same shape

        for i, (text, image) in enumerate(zip(texts, self._fixed_images)):
            # classify, output is a tupe (index, score)
            ind, score = self.model.classify(image, text).values()
            out[i][ind] = score
            out[i][1 - ind] = 1 - score

        return out

    def explain(self, images: np.ndarray, texts: np.ndarray, mode: str):
        """ Main API to calculate shap values

        Args:
            images: np.ndarray of shape (N, D1, D2, C);
                N = number of samples
                D1, D2, C = three channel image
            texts: np.ndarray of shape (N,)

        Returns:
            shap values calculated

        """

        # input validations
        if mode not in self._supported_modes:
            raise ValueError(f"This mode {mode} is not supported!")

        if images.shape[0] != texts.shape[0]:
            raise ValueError(f"Shape mismatch, inputs' first dimensions should be equal!")

        if mode == "text_only":
            self._fixed_images = images
            if not isinstance(images[0], Image.Image):
                self._fixed_images = utils.arr_to_img(images)
            # tokenizer and masker
            tokenizer = AutoTokenizer.from_pretrained(
                "bert-base-uncased", use_fast=True)
            text_masker = shap.maskers.Text(tokenizer)
            # NOTE: if using text heatmap you need output_names arg to let it know its text output!!!
            explainer = shap.Explainer(
                self._f_text, text_masker, algorithm=self.algorithm)
            shap_values = explainer(
                texts, max_evals=self.max_evals, batch_size=self.batch_size)

        elif mode == "image_only":
            self._fixed_texts = texts
            image_masker = shap.maskers.Image("inpaint_telea", images[0].shape)
            image_explainer = shap.Explainer(
                self._f_image, image_masker, algorithm=self.algorithm)
            shap_values = image_explainer(
                images, max_evals=self.max_evals, batch_size=self.batch_size)

        return shap_values

    def text_plot(self, shap_values):
        """ plot the text chart given shap values

        Args:
            shap_values: 
        """

        # NOTE: to use text force plot the arg shap_values (and .base_value) has to be 1-d
        # (i.e. (#samples, )) e.g. get only the hateful case, we shrink shap to 1-d array
        # note how opposite to image, where a 2-d array is needed (shape = #samples, 1)
        shap_values = copy.deepcopy(shap_values)
        shap_values = shap_values[..., 1]
        return modified_text_plot(shap_values)
        # shap.text_plot(shap_values)

    def image_plot(self, shap_values):
        """ plot the image chart given shap values

        Args:
            shap_values: 
        """

        shap_values = copy.deepcopy(shap_values)
        # have to change it back to float if getting 'true_divide' error
        shap_values.data = shap_values.data.astype(np.float64)

        # NOTE: image plot needs shap value (base_values) to be 2-d (# samples * # model outputs)
        # in the case of a single sample we need an array of (1, 1)
        shap_values = shap_values[..., 1:2]
        return image(shap_values, labels=[['hateful']])


def _examples():
    """ Example for how to use this explainer
    """
    # read data to try
    data_path = r"hm-data/"
    labels = utils.read_labels(data_path + "train.jsonl", True)
    ids = [28061]
    target_labels = [l for l in labels if l['id'] in ids]
    print(f" target_labels = {target_labels}")
    target_images, target_texts = utils.parse_labels(
        target_labels, img_to_array=True, separate_outputs=True)

    # model to explain
    model = MMBT.from_pretrained("mmbt.hateful_memes.images")

    # Explainer hyper params
    max_evals = 100
    batch_size = 50
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # test default partition algo
    explainer = Explainer(model, max_evals=max_evals, batch_size=batch_size)
    # text_shap_values = explainer.explain(target_images, target_texts, "text_only")
    image_shap_values = explainer.explain(target_images, target_texts, "image_only")

    # plots
    # explainer.text_plot(text_shap_values) 
    explainer.image_plot(image_shap_values)
    


if __name__ == "__main__":
    _examples()
