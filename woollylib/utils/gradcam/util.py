import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

from woollylib.utils.gradcam.gradcam import GradCAM


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


def apply_heatmap_to_image(img, heatmap):
    # Read original image form path
    img = img.cpu().numpy().transpose(1, 2, 0)

    # Resize heat map to original image
    heatmap = cv.resize(heatmap.numpy(), (img.shape[1], img.shape[0]))
    
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv.applyColorMap(heatmap, cv.COLORMAP_JET)/255.0

    # Superimpose heat map on original image
    superimposed_img = heatmap * 0.8 + img

    # Normalize super imposed image
    return superimposed_img / superimposed_img.max()


def plot_output(image, heatmap, simage, label, pred, class_map):
    # Get classes names
    classes = list(class_map.keys())

    # Define figure
    fig = plt.figure(figsize=(6, 4))
    
    ax = fig.add_subplot(1, 3, 1, xticks=[], yticks=[])
    ax.set_title(f'{classes[label.item()]}/{classes[pred.item()]}')
    plt.imshow(image.cpu().numpy().transpose(1, 2, 0))
    
    ax = fig.add_subplot(1, 3, 2, xticks=[], yticks=[])
    ax.set_title(f'{classes[label.item()]}/{classes[pred.item()]}')
    plt.imshow(heatmap.squeeze())

    ax = fig.add_subplot(1, 3, 3, xticks=[], yticks=[])
    ax.set_title(f'{classes[label.item()]}/{classes[pred.item()]}')
    plt.imshow(simage)
