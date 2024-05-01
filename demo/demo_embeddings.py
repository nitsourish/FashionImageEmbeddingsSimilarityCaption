import gzip
import os
import pickle
# %pip install opencv-python
import random
import urllib.request
from itertools import cycle
from typing import Any, Dict, List, Tuple

import numpy as np
import requests
import streamlit as st
import torch
from img2vec_pytorch import Img2Vec
from PIL import Image
from tqdm import tqdm

sideBar = st.sidebar
model_name = sideBar.selectbox(
    "Which pretrained model do you want to use?",
    ("densenet", "alexnet", "vgg", "efficientnet", "resnet"),
)

if model_name == "resnet":
    sideBar = st.sidebar
    model_name = sideBar.selectbox(
        "Which pretrained resnet model do you want to use?",
        ("resnet18", "resnet34", "resnet50", "resnet152"),
    )

if model_name == "efficientnet":
    sideBar = st.sidebar
    suffix = sideBar.selectbox(
        "Which pretrained resnet model do you want to use?",
        ("b2", "b1", "b3", "b4", "b5", "b7"),
    )
    model_name = model_name + "_" + suffix

gen_kwargs = {"model": model_name}


# docstring and type hints
def embedding_write(predicted_embeddings: dict):
    """This function writes the embeddings to a pickle file"""

    with open("image_vec_dict.pickle", "wb") as handle:
        pickle.dump(
            predicted_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL
        )
        # display the message to the user
    st.write("Embeddings Generated and saved Successfully!")


def show_sample_images():
    """This function displays the sample images and their embeddings"""

    filteredImages = {"First": "1559.jpg", "Second": "palace.jpg"}

    cols = cycle(st.columns(3))
    for filteredImage in filteredImages.values():
        next(cols).image(filteredImage, width=200)
    for i, filteredImage in enumerate(filteredImages.values()):
        if next(cols).button("Embeddings Generation", key=i):
            embedding = predict_oneshot(filteredImage, False)
            st.write(embedding)


def image_uploader():
    """This function takes the image files from the user and displays the embeddings"""

    with st.form("uploader"):
        images = st.file_uploader(
            "Embeddings Generation",
            accept_multiple_files=True,
            type=["png", "jpg", "jpeg"],
        )
        images = [image.name for image in images]
        submitted = st.form_submit_button("Submit")
        if submitted:
            predicted_embeddings = predict_batch(images, False)
            embedding_write(predicted_embeddings)
            for image, embedding in predicted_embeddings.items():
                st.write(image)
                st.write(embedding)
                st.write("---------------------------------")


def images_url():
    """This function takes the image links/urls from the user and displays the embeddings"""

    with st.form("url"):
        urls = st.text_input(
            "Enter URL of Images followed by comma for multiple URLs"
        )
        images = urls.split(",")
        submitted = st.form_submit_button("Submit")
        if submitted:
            predicted_embeddings = predict_batch(images, True)
            embedding_write(predicted_embeddings)
            for image, embedding in predicted_embeddings.items():
                st.write(image)
                st.write(embedding)
                st.write("---------------------------------")


def main():
    """This is the main function that runs the streamlit app"""

    st.title("Image Embeddings Generation")
    st.header("Welcome to Image Embeddings Generation!")
    st.write(
        "This is a sample app that demonstrates Embedding Generation for fashion Images.🚀"
    )
    st.write(
        "Visit the [Github](https://github.com/nitsourish/FashionImageEmbeddings.git) repo for detailed exaplaination and to get started right away"
    )
    tab1, tab2, tab3 = st.tabs(
        ["Sample Images", "Image from computer", "Image from URL"]
    )
    with tab1:
        show_sample_images()
    with tab2:
        image_uploader()
    with tab3:
        images_url()


# type hints and docstring
def predict_batch(images_list: List, is_url: bool) -> dict:
    """This function takes the list of images and returns the embeddings of the images"""

    images = []
    for image in tqdm(images_list):
        if is_url:
            urllib.request.urlretrieve(image, "file.jpg")
            i_image = Image.open("file.jpg")

        else:
            i_image = Image.open(image)

        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

        images.append(i_image)

    img2vec = Img2Vec(**gen_kwargs)
    vec = img2vec.get_vec(images, tensor=False)
    image_vec_dict = dict(zip(images_list, vec))
    if is_url:
        os.remove("file.jpg")
    return image_vec_dict


def predict_oneshot(image: Any, is_url: bool) -> np.ndarray:
    """This function takes the image and returns the embeddings of the image"""

    if is_url:
        urllib.request.urlretrieve(image, "file.jpg")
        i_image = Image.open("file.jpg")

    else:
        i_image = Image.open(image)

    if i_image.mode != "RGB":
        i_image = i_image.convert(mode="RGB")

    img2vec = Img2Vec(**gen_kwargs)
    vec = img2vec.get_vec(i_image, tensor=False)
    if is_url:
        os.remove("file.jpg")
    return vec


if __name__ == "__main__":
    main()
