import gzip
import os
import pickle
# %pip install opencv-python
import random
import urllib.request
from itertools import cycle
from typing import Any, Dict, List, Tuple, Union
import numpy as np
import pandas as pd
import requests
import streamlit as st
import torch
# from image_to_vec import Img2Vec
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

sideBar = st.sidebar
similarity_metric = sideBar.selectbox(
    "Which distance metric do you want to measure similarity?",
    ("cosine similarity", "dot-product", "pearson correlation"),
)

model_name = "resnet"
if model_name == "resnet":
    sideBar = st.sidebar
    model_name = sideBar.selectbox(
        "Which pretrained resnet model do you want to use?",
        ("resnet18", "resnet34"),
    )


if similarity_metric == "cosine similarity":
    distance_metric = "cosine similarity"
elif similarity_metric == "pearson correlation":
    distance_metric = "pearson"
else:
    distance_metric = "dot-product"

gen_kwargs = {"model": model_name}

with open("demo/image_to_vec.py") as file:
    exec(file.read())

def precomputed_embedding_load() -> dict:
    """This function loads the embeddings from a pickle file"""

    with open("image_vec_dict_all.pkl", "rb") as handle:
        embeddings = pickle.load(handle)
        # display the message to the user
    st.write("Embeddings loaded Successfully!")
    return embeddings


def show_sample_images():
    """This function displays the sample images and its similar images"""

    filteredImages = {"First": "test_image5.jpg"}
    image_vec_dict = precomputed_embedding_load()
    cols = cycle(st.columns(3))
    for filteredImage in filteredImages.values():
        next(cols).image(filteredImage, width=200)

    for i, filteredImage in enumerate(filteredImages.values()):
        if next(cols).button("Similar Fashion Generation"):
            similar_images, similarity_score, img, df = find_similar_images(
                filteredImage, image_vec_dict, top_n=3
            )
            st.write("Similar Images:", similar_images)
            st.write("Similarity score:", similarity_score)
            st.write(df)
            img.show()


def image_uploader():
    """This function takes the image files from the user and displays similar images"""

    with st.form("uploader"):
        images = st.file_uploader(
            "Similar Fashion Generation",
            accept_multiple_files=True,
            type=["png", "jpg", "jpeg"],
        )
        images = [image.name for image in images]
        images = ['./images/sample_images/' + image for image in images] 
        submitted = st.form_submit_button("Submit")
        if submitted:
            df = find_similar_images_multiple(
                images, precomputed_embedding_load(), top_n=3
            )
            st.write(df)


def images_url():
    """This function takes the image links/urls from the user and displays similar images"""

    with st.form("url"):
        urls = st.text_input(
            "Enter URL of Images followed by comma for multiple URLs"
        )
        images = urls.split(",")
        submitted = st.form_submit_button("Submit")
        if submitted:
            df = find_similar_images(
                urls, precomputed_embedding_load(), top_n=3
            )
            st.write(df)


def main():
    """This is the main function that runs the streamlit app"""

    st.title("Similar Fashion Generation")
    st.header("Welcome to Similar Fashion Generation!")
    st.write(
        "This is a sample app that demonstrates Similar Fashion Generation for fashion Images.🚀"
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


# type hints and docstring
def image_similarity_calculation(
    img: Any, image_vec_dict: dict, top_n: int = 3
) -> Tuple[List[str], List[float], Any]:
    """This function calculates the similarity between the source image and the images in the dataset"""

    img2vec = Img2Embebedding(**gen_kwargs)
    vec = img2vec.get_vec(img, tensor=False)
    similarity_dict = {}
    for key, value in tqdm(image_vec_dict.items()):
        if distance_metric == "cosine similarity":
            similarity_dict[key] = cosine_similarity(
                vec.reshape(1, -1), value.reshape(1, -1)
            )[0][0]
        elif distance_metric == "dot-product":
            similarity_dict[key] = np.dot(vec, value) / (
                np.linalg.norm(vec) * np.linalg.norm(value)
            )
        else:
            similarity_dict[key] = np.corrcoef(vec, value)[0][1]
    similarity_dict = dict(
        sorted(similarity_dict.items(), key=lambda x: x[1], reverse=True)
    )
    similar_images = list(similarity_dict.keys())[1 : top_n + 1]
    similarity_score = list(similarity_dict.values())[1 : top_n + 1]
    for img in similar_images:
        img = Image.open(img)
        img.show()
    return similar_images, similarity_score, img


def find_similar_images(
    image_path, image_vec_dict: dict, top_n: int = 3
) -> Tuple[List[str], List[float], Any, pd.DataFrame]:
    """This function finds the top3 similar images for the source image"""

    img = Image.open(image_path).convert("RGB")
    img.show()
    similar_images, similarity_score, img = image_similarity_calculation(
        img, image_vec_dict, top_n
    )
    # create a dataframe with source image name,top 3 similar images and similarity score
    similar_images = [os.path.basename(img) for img in similar_images]
    image_path = os.path.basename(image_path)
    df = pd.DataFrame(
        {
            "source_image": [image_path],
            "similar_image_1": similar_images[0],
            "similarity_score_1": similarity_score[0],
            "similar_image_2": similar_images[1],
            "similarity_score_2": similarity_score[1],
            "similar_image_3": similar_images[2],
            "similarity_score_3": similarity_score[2],
        }
    )
    return similar_images, similarity_score, img, df


# make the function works for list of source images
def find_similar_images_multiple(
    image_paths: List[str], image_vec_dict, top_n=3
) -> pd.DataFrame:
    """This function finds the top3 similar images for the multiple source images"""

    df = pd.DataFrame()
    for image_path in image_paths:
        _, _, _, df_temp = find_similar_images(
            image_path, image_vec_dict, top_n
        )
        df = pd.concat([df, df_temp])

    return df


if __name__ == "__main__":
    main()
