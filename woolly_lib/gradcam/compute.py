# Create dataset to load images

from torch.utils import data

from woolly_lib.dataset import WyCustomDataset
from woolly_lib.transform import BASE_PROFILE, get_transform
from woolly_lib.gradcam.gradcam import GradCAM
from woolly_lib.gradcam.util import get_prediction_for_image, generate_heat_map, apply_heatmap_to_image, plot_output


def compute_gradcam(model, class_map, path='./data', device='cpu'):
    # use the CIFAR transformation
    profile = {
        'resize': BASE_PROFILE['resize'],
        'normalize': BASE_PROFILE['normalize'],
        'to_tensor': BASE_PROFILE['to_tensor']
    }

    transform = get_transform(profile)

    # define a local image dataset
    dataset = WyCustomDataset(class_map, path=path, transforms=transform)

    # define the dataloader to load that single image
    dataloader = data.DataLoader(dataset=dataset, shuffle=False, batch_size=1)

    # initialize the VGG model
    gradcam = GradCAM(model).to(device)

    for img, label, (impath,) in dataloader:
        # To device
        img = img.to(device)
        # Get Predictions
        prediction = get_prediction_for_image(gradcam, img, device)
        # Get Heatmap for this prediction
        heatmap = generate_heat_map(gradcam, prediction, img, channel_size=128)
        # Get Overlayed image
        superimposed = apply_heatmap_to_image(impath, heatmap)

        # Plot Images
        plot_output(heatmap, superimposed, label, class_map)

