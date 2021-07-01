import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

from woolly_lib.gradcam.gradcam import GradCAM


def get_prediction_for_image(gradcam: GradCAM, data, device: str):
    # set the evaluation mode
    gradcam.eval()

    # Convert data to device
    data = data.to(device)

    # get the most likely prediction of the model
    prediction = gradcam(data)  # .argmax(dim=1)

    return prediction


def generate_heat_map(gradcam: GradCAM, prediction, data, channel_size: int = 128):
    # get the gradient of the output with respect to the parameters of the model
    prediction[:, prediction.argmax(dim=1).item()].backward()

    # pull the gradients out of the model
    gradients = gradcam.get_activations_gradient()

    # pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    # get the activations of the last convolutional layer
    activations = gradcam.get_activations(data).detach()

    # weight the channels by corresponding gradients
    for i in range(channel_size):
        activations[:, i, :, :] *= pooled_gradients[i]

    # average the channels of the activations
    heatmap = torch.mean(activations, dim=1).squeeze()

    # relu on top of the heatmap
    # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
    heatmap = np.maximum(heatmap.cpu(), 0)

    # normalize the heatmap
    heatmap /= torch.max(heatmap)

    return heatmap


def apply_heatmap_to_image(path: str, heatmap):
    # Read original image form path
    img = cv.imread(path)

    # Resize heat map to original image
    heatmap = cv.resize(heatmap.numpy(), (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv.applyColorMap(heatmap, cv.COLORMAP_JET)

    # Superimpose heat map on original image
    superimposed_img = heatmap * 0.4 + img

    # Normalize super imposed image
    return superimposed_img / superimposed_img.max()


def plot_output(heatmap, image, label, class_map):
    # Get classes names
    classes = list(class_map.keys())

    # Define figure
    fig = plt.figure(figsize=(10, 4))

    ax = fig.add_subplot(1, 2, 1, xticks=[], yticks=[])
    ax.set_title(f'{classes[label.item()]}')
    plt.imshow(image)

    ax = fig.add_subplot(1, 2, 2, xticks=[], yticks=[])
    ax.set_title(f'{classes[label.item()]}')
    plt.imshow(heatmap.squeeze())
