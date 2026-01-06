import numpy as np
import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries

def predict_wrapper(model):
    """
    Wraps model.predict so LIME can call it.
    Input: (N, 224, 224, 3)
    Output: probabilities
    """
    def _predict(images):
        return model.predict(images)
    return _predict


def compute_lime(model, img_tensor, class_idx):
    """
    img_tensor: (1, 224, 224, 3)
    returns: LIME visualization image
    """
    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(
        img_tensor[0],
        classifier_fn=predict_wrapper(model),
        top_labels=2,
        hide_color=0,
        num_samples=1000
    )

    temp, mask = explanation.get_image_and_mask(
        label=class_idx,
        positive_only=True,
        num_features=8,
        hide_rest=False
    )

    lime_img = mark_boundaries(temp, mask)
    return lime_img