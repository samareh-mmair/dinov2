import os
import sys
import time

import urllib
import urllib.request as request

import io
import numpy as np
from PIL import Image
import cv2

def load_array_from_url(url: str) -> np.ndarray:
    with urllib.request.urlopen(url) as f:
        array_data = f.read()
        g = io.BytesIO(array_data)
        return np.load(g)


def load_image_from_url(url: str) -> Image:
    with request.urlopen(url) as f:
        return Image.open(f).convert("RGB")

def load_image_from_path(file_path: str) -> Image:
    return Image.open(file_path).convert("RGB")

def load_resize_image_from_path(file_path: str) -> Image:
    img = Image.open(file_path).convert("RGB")
    img = img.resize((640, 400), Image.Resampling.LANCZOS)
    return img
    
# Precomputed foreground / background projection
STANDARD_ARRAY_URL = "https://dl.fbaipublicfiles.com/dinov2/arrays/standard.npy"
standard_array = load_array_from_url(STANDARD_ARRAY_URL)

# EXAMPLE_IMAGE_URL = "https://dl.fbaipublicfiles.com/dinov2/images/example.jpg"
# example_image = load_image_from_url(EXAMPLE_IMAGE_URL)
# EXAMPLE_IMAGE_PATH1 = "/media/nas/AIR/Team/Samareh/Stitch_images/Data/EODE/0010_hyd_aktogay_trimmed/007/images/0001385.jpg"
# EXAMPLE_IMAGE_PATH2 = "/media/nas/AIR/Team/Samareh/Stitch_images/Data/EODE/0010_hyd_aktogay_trimmed/007/images/0001388.jpg"
EXAMPLE_IMAGE_PATH1 = "/media/nas/AIR/Team/Samareh/DINO/elephant1.png"
EXAMPLE_IMAGE_PATH2 = "/media/nas/AIR/Team/Samareh/DINO/elephant2.png"
example_image1 = load_resize_image_from_path(EXAMPLE_IMAGE_PATH1)
example_image2 = load_resize_image_from_path(EXAMPLE_IMAGE_PATH2)

disparity1 = load_image_from_path('disparity85.jpg')
disparity1 = disparity1.point( lambda p: 1 if p > 140 else 0 )
example_image1 = Image.fromarray(np.array(disparity1)*np.array(example_image1).astype(np.uint8))

disparity2 = load_image_from_path('disparity88.jpg')
disparity2 = disparity2.point( lambda p: 1 if p > 140 else 0 )
example_image2 = Image.fromarray(np.array(disparity2)*np.array(example_image2).astype(np.uint8))
example_image1.save('example_image1.png',format='png')
example_image2.save('example_image2.png',format='png')
# display(example_image)

from typing import Tuple

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from scipy.ndimage import binary_closing, binary_opening
import torch
import torchvision.transforms as transforms


REPO_NAME = "facebookresearch/dinov2"
MODEL_NAME = "dinov2_vitb14"


DEFAULT_SMALLER_EDGE_SIZE = 448
DEFAULT_BACKGROUND_THRESHOLD = 0.05
DEFAULT_APPLY_OPENING = False
DEFAULT_APPLY_CLOSING = False


def make_transform(smaller_edge_size: int) -> transforms.Compose:
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
    interpolation_mode = transforms.InterpolationMode.BICUBIC

    return transforms.Compose([
        transforms.Resize(size=smaller_edge_size, interpolation=interpolation_mode, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])


def prepare_image(image: Image,
                  smaller_edge_size: float,
                  patch_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    transform = make_transform(int(smaller_edge_size))
    image_tensor = transform(image)

    # Crop image to dimensions that are a multiple of the patch size
    height, width = image_tensor.shape[1:] # C x H x W
    cropped_width, cropped_height = width - width % patch_size, height - height % patch_size
    image_tensor = image_tensor[:, :cropped_height, :cropped_width]

    grid_size = (cropped_height // patch_size, cropped_width // patch_size) # h x w (TODO: check)
    return image_tensor, grid_size


def make_foreground_mask(tokens,
                         grid_size: Tuple[int, int],
                         background_threshold: float = 0.0,
                         apply_opening: bool = True,
                         apply_closing: bool = True):
    projection = tokens @ standard_array
    mask = projection > background_threshold
    mask = mask.reshape(*grid_size)
    if apply_opening:
        mask = binary_opening(mask)
    if apply_closing:
        mask = binary_closing(mask)
    return mask.flatten()


def render_patch_pca(image: Image,
                     smaller_edge_size: float = 448,
                     patch_size: int = 14,
                     background_threshold: float = 0.05,
                     apply_opening: bool = False,
                     apply_closing: bool = False) -> Image:
    image_tensor, grid_size = prepare_image(image, smaller_edge_size, patch_size)
    device = 'cuda'
    
    with torch.inference_mode():
        image_batch = image_tensor.unsqueeze(0)
        start = time.time()
        tokens = model.get_intermediate_layers(image_batch)[0].squeeze() #.to(device)
        print(f'runtime: {time.time()-start}')

    # mask = make_foreground_mask(tokens,
    #                             grid_size,
    #                             background_threshold,
    #                             apply_opening,
    #                             apply_closing)

    pca = PCA(n_components=3)
    # pca.fit(tokens[mask])
    pca.fit(tokens)
    projected_tokens = pca.transform(tokens)

    t = torch.tensor(projected_tokens)
    t_min = t.min(dim=0, keepdim=True).values
    t_max = t.max(dim=0, keepdim=True).values
    normalized_t = (t - t_min) / (t_max - t_min)
    print(t.max())
    array = (normalized_t * 255).byte().numpy()
    # array[~mask] = 0
    array = array.reshape(*grid_size, 3)

    return Image.fromarray(array).resize((image.width, image.height), 0)



print(f"using {MODEL_NAME} model")
device = 'cuda'
model = torch.hub.load(repo_or_dir=REPO_NAME, model=MODEL_NAME) #.to(device)
model.eval()
print(f"patch size: {model.patch_size}")

image_patch1 = render_patch_pca(image=example_image1,
                 smaller_edge_size=DEFAULT_SMALLER_EDGE_SIZE,
                 patch_size=model.patch_size ,
                 background_threshold=DEFAULT_BACKGROUND_THRESHOLD,
                 apply_opening=DEFAULT_APPLY_OPENING,
                 apply_closing=DEFAULT_APPLY_CLOSING)

image_patch1.save('patch1.png',format='png')

image_patch2 = render_patch_pca(image=example_image2,
                 smaller_edge_size=DEFAULT_SMALLER_EDGE_SIZE,
                 patch_size=model.patch_size,
                 background_threshold=DEFAULT_BACKGROUND_THRESHOLD,
                 apply_opening=DEFAULT_APPLY_OPENING,
                 apply_closing=DEFAULT_APPLY_CLOSING)

image_patch2.save('patch2.png',format='png')
# if INSTALL: # Try installing package
#     !{sys.executable} -m pip install -U ipywidgets

# Initiate ORB detector
orb = cv2.ORB_create()
# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(np.asarray(image_patch1),None)
kp2, des2 = orb.detectAndCompute(np.asarray(image_patch2),None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = bf.match(des1,des2)


# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
# Draw first 10 matches.
img3 = cv2.drawMatches(np.asarray(example_image1),kp1,np.asarray(example_image2),kp2,matches[:30],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite('keys.jpg',img3)
import ipywidgets as widgets


def get_image(image_data: bytes) -> Image:
    with io.BytesIO(image_data) as f:
        return Image.open(f).convert("RGB")


def get_image_data(image: Image) -> bytes:
    with io.BytesIO() as f:
        image.save(f, format="PNG")
        return f.getvalue()


def make_blank_image(size: Tuple[int, int]) -> Image:
    return Image.new(mode="RGBA", size=size, color=(0, 0, 0, 0))


blank_result = make_blank_image(size=example_image1.size)

DEFAULT_IMAGE_DATA = get_image_data(example_image1)
DEFAULT_RESULT_DATA = get_image_data(blank_result)

image_widget = widgets.Image(
    value=DEFAULT_IMAGE_DATA,
    width=512,
    height=384,
)
upload_widget = widgets.FileUpload(
    accept="image/*",
    multiple=False,
)
smaller_edge_size_widget = widgets.IntSlider(
    value=DEFAULT_SMALLER_EDGE_SIZE,
    min=84,
    max=1344,
    step=1,
    description="Image size (smaller edge):",
    disabled=False,
    continuous_update=False,
    orientation="horizontal",
    readout=True,
    readout_format="d",
    style={"description_width": "initial"},
    layout=widgets.Layout(width="50%"),
)
background_threshold_widget = widgets.FloatSlider(
    value=DEFAULT_BACKGROUND_THRESHOLD,
    min=-1.0,
    max=1.0,
    step=0.01,
    description="Background threshold:",
    disabled=False,
    continuous_update=False,
    orientation="horizontal",
    readout=True,
    readout_format="0.02f",
    style={"description_width": "initial"},
    layout=widgets.Layout(width="50%"),
)
apply_opening_widget = widgets.Checkbox(
    value=DEFAULT_APPLY_OPENING,
    description="Apply opening operation",
    indent=False,
)
apply_closing_widget = widgets.Checkbox(
    value=DEFAULT_APPLY_CLOSING,
    description="Apply closing operation",
    indent=False,
)
clear_widget = widgets.Button(
    description="Clear",
    disabled=False,
    button_style="",
    tooltip="Click to reset inputs",
    icon="",
)
submit_widget = widgets.Button(
    description="Submit",
    disabled=False,
    button_style="success",
    tooltip="Click to run with specified inputs",
    icon="check",
)
result_widget = widgets.Image(
    value=DEFAULT_RESULT_DATA,
    width=512,
    height=384,
)

image_hbox = widgets.HBox([image_widget, result_widget])
button_hbox = widgets.HBox([upload_widget, clear_widget, submit_widget])
settings_vbox = widgets.VBox([
    smaller_edge_size_widget,
    background_threshold_widget,
    apply_opening_widget,
    apply_closing_widget,
])

box_widget = widgets.VBox(children=[
    image_hbox,
    settings_vbox,
    button_hbox,
])


def upload_callback(widget):
    print("Updating image")

    image_data = upload_widget.value[0]["content"]
    image_widget.value = image_data

    image = get_image(image_data)
    blank_image = make_blank_image(image.size)
    result_data = get_image_data(blank_image)
    result_widget.value = result_data


def clear_callback(widget):
    print("Resetting inputs")

    image_widget.value = DEFAULT_IMAGE_DATA
    smaller_edge_size_widget.value = DEFAULT_SMALLER_EDGE_SIZE
    background_threshold_widget.value = DEFAULT_BACKGROUND_THRESHOLD
    apply_opening_widget.value = DEFAULT_APPLY_OPENING
    apply_closing_widget.value = DEFAULT_APPLY_CLOSING
    result_widget.value = DEFAULT_RESULT_DATA


def submit_callback(widget):
    print("Rendering PCA")

    image = get_image(image_widget.value)
    print(image.size)
    result = render_patch_pca(image=image,
                              smaller_edge_size= smaller_edge_size_widget.value ,
                              patch_size=model.patch_size,
                              background_threshold=background_threshold_widget.value,
                              apply_opening=apply_opening_widget.value,
                              apply_closing=apply_closing_widget.value)
    result_data = get_image_data(result)
    result_widget.value = result_data


upload_widget.observe(upload_callback, names="value")
clear_widget.on_click(clear_callback)
submit_widget.on_click(submit_callback)

# display(box_widget)