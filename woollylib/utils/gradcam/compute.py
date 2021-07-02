# Create dataset to load images

import torch
from torch.utils import data

from woollylib.utils.gradcam.gradcam import GradCAM
from woollylib.utils.gradcam.util import get_prediction_for_image, generate_heat_map, apply_heatmap_to_image, plot_output


def compute_gradcam(model, class_map, img, label, pred, device='cpu'):
    # initialize the GradCAM model
    gradcam = GradCAM(model).to(device)

    # To device
    img = img.to(device)
    # Get Predictions
    prediction = get_prediction_for_image(gradcam, torch.reshape(img, (1, 3, 32, 32)), device)
    # Get Heatmap for this prediction
    heatmap = generate_heat_map(gradcam, prediction, torch.reshape(img, (1, 3, 32, 32)), channel_size=256)
    # Get Overlayed image
    superimposed = apply_heatmap_to_image(img, heatmap)

    # Plot Images
    plot_output(img, heatmap, superimposed, label, pred, class_map)