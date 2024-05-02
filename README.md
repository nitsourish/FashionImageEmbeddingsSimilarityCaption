# FashionImageEmbeddingSimilarityCaption

This repository leverages pre-trained models in PyTorch to extract vector embeddings for any image and find similar images.

## Purpose of the Repository
1) The purpose of this repository is to provide scripts and demonstrations for generating embeddings for fashion images.
2) It also includes functionality to find similar fashion images based on the extracted embeddings and pre-computed embeddings.
3) Additionally, it can generate captions for any given image using pre-trained models.


## Pointwise Content of the Repository
The repository contains the following files and directories:

- `image_to_vec.py`: Source scripts for invoking pre-trained models in PyTorch 
- `demo_embeddings.py`: This script is used to extract vector embeddings for fashion images using the pre-trained models.
- `demo_similarity.py`: This script allows finding similar fashion images based on the extracted embeddings.
- `demo_image_caption.py`: This script uses the pre-trained models to generate captions for any given image.
- `README.md`: This file provides an overview of the repository and its purpose.
- `image_vec_dict_all.pkl`: pre-computed embeddings
- `images`: sample fashion images of 8888(source https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset)

Feel free to explore the repository and use the provided scripts to extract embeddings, find similar images, and generate captions for fashion images.


## Streamlit Implementation
To provide an interactive playground for exploring multiple pre-trained models and different similarity measures, we have implemented a Streamlit application.

The Streamlit application allows users to upload their own fashion images and select from a variety of pre-trained models. They can then choose different similarity measures to find similar fashion images based on the extracted embeddings.

The implementation includes the following features:
- Upload functionality: Users can upload their own fashion images to be processed by the selected pre-trained model.
- Pre-trained model options: Users can choose from a list of pre-trained models to extract embeddings from the uploaded images.
- Similarity measure options: Users can select different similarity measures to find similar fashion images based on the extracted embeddings.
- Interactive visualization: The application provides an interactive visualization of the uploaded image and the similar images found.

By using this Streamlit implementation, users can easily experiment with different pre-trained models and similarity measures to explore the capabilities of the repository and gain insights into fashion image embeddings.

Additionally implemention of an image caption generation application using a pre-trained Transformer architecture with the Streamlit framework, allowing user flexibility in controlling the caption length:

## Streamlit Handlers

### Embeddings

| Model           | Embedding Dim |
|-----------------|---------------|
| `Resnet-18`     | 512           |
| `Resnet-34`     | 512           |
| `Resnet-50`     | 2048          |
| `Resnet-152`    | 2048          |
| `alexnet`       | 4096          |
| `vgg-11`        | 4096          |
| `densenet`      | 1024          |
| `efficientnetb1`| 1024          |
| `efficientnetb2`| 1408          |
| `efficientnetb3`| 1536          |
| `efficientnetb4`| 1792          |
| `efficientnetb5`| 2048          |
| `efficientnetb7`| 2560          |

### Other Handlers

| App           | User Control        |            Variants               |
|---------------|---------------------|-----------------------------------|
| `Similarity`  | Similarity Measure  |Cosine,dot-prod,pearson correlation|
| `ImageCaption`| Beam Search Param   |       4,5                         |
| `ImageCaption`| Length              |        32                         |


## How to run

To run the Streamlit application, follow these steps:

1. Clone the repository by running the following command in your terminal:
    ```
    git clone https://github.com/nitsourish/FashionImageEmbeddingsSimilarityCaption.git
    ```

2. Navigate to the repository directory:
    ```
    cd FashionImageEmbeddingsSimilarityCaption
    ```

3. Create a virtual environment to install the required dependencies:
    ```
    python -m venv venv
    ```

4. Activate the virtual environment:
    - For Windows:
      ```
      venv\Scripts\activate
      ```
    - For macOS/Linux:
      ```
      source venv/bin/activate
      ```

5. Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```

6. Run the Streamlit application by executing the following commands:
    ```
    streamlit run demo_embeddings.py
    streamlit run demo_similarity.py
    streamlit run demo_image_caption.py
    ```

7. Live Streamlit cloud Apps can be fired from here
   ```
   https://fashionimageembeddingssimilaritycaption-q7wwrxgt84itn8zhkwsxap.streamlit.app/
   https://fashionimageembeddingssimilaritycaption-njhpbpgnf7brd8f8o54r6t.streamlit.app/
   https://fashionimageembeddingssimilaritycaption-pjgl4ixyaqnqwdv8ahzjm6.streamlit.app/
   ```

Now you can explore the Streamlit application and use the provided scripts to extract embeddings, find similar images, and generate captions for fashion images.




