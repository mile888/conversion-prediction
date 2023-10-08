# Probability conversion prediction
Over the past decade, digital marketing has seen significant growth and development. However, the accuracy of prediction technologies in this field has often fallen short, particularly when it comes to predicting conversions. Conversion prediction remains a challenging task for many marketing professionals. In this task, we leverage a dataset that includes the content of landing pages, the number of impressions made by users, user features, as well as the number of conversions made by users after seeing advertisement on impressions. 

## Task
The goal of this task is to accurately predict the probability of a user making a conversion for each impression of a certain advertising banner. To solve this task, we consider this problem as a classification task determined by a conversion rate. As a result, we consider the probability outputs of the classifier. We assume that the results of the conversion probability are more accurate for the classifier with a better F1-score. 

## Installation
Clone this repository:
```
git clone https://github.com/mile888/conversion-prediction.git
```
The required packages can be installed via standard Python package manager:
```
pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset
The datasets for this task are available [here](https://drive.google.com/drive/folders/1eH5FLgAMjcMGdd8yHxjnO5IJOOZdJIuv).

The dataset consists of three files:
1. **ds_homework_sample.csv** - contains a random sample of the work of an advertising network with approximately 100 sites over two weeks. Each line represents data from the conversions and impressions metrics aggregated over the rest attributes. 
2. **train.csv** - is a cleaned `ds_homework_sample.csv` file prepared for classifier's training.
3. **test.csv** - is a cleaned `ds_homework_sample.csv` file prepared for classifier's validation.


## Guideline
Tree of the project:
```plaintext
project_root/
├── .github/workflows/ 
│   └── main.yaml
├── figures/
│   ├── best_features.png
│   ├── full_plot.png
│   └── undersample_plot.png  
├── save_models/
│   ├── best_features.pickle
│   ├── knn_model.pickle
│   ├── knn_param.pickle
│   ├── lr_model.pickle
│   ├── lr_param.pickle
│   ├── rf_model.pickle
│   ├── rf_param.pickle
│   ├── xgb_model.pickle
│   └── xgb_param.pickle
├── source/
│   ├── __init__.py
│   ├── mrmr.py
│   ├── predict.py
│   ├── train_validate.py
│   └── utils.py
├── .gitignore 
├── config.yaml
├── main.ipynb
├── main.py
├── README.md
└── requirements.txt
```
- [x] **To run this program**, you should call in the terminal:
```
python main.py
```
- [x] This repository contains configuration file `config.yaml` where you can 
chose the mode: 
1. `train` - it means to train classifier on `train.csv` dataset and validate on `test.csv` dataset.
2. `predict` - it means used trained models and directly work with raw dataset, for example `ds_homework_sample.csv` in this case.

- Accordingly you can config the roots of the datasets and chose the next options:
1. `train.csv` dataset - (e.g. `dataset_root_train: "datasets/train.csv"`)
2. `test.csv` dataset - (e.g. `dataset_root_test: "datasets/test.csv"`)
3. model (classifier) - (e.g. `model: "RF"`)
4. k best features - (e.g. `k_best_features: 2`)
5. 

For example: 
- the root of the image dataset is:
```
root_images: "dataset/image-db/"
```
- query image is located in:
```
root_query_image: "dataset/test-cases/cat.jpg"
```
and the name of the collage image is:
```
save_figure_name: "cat.jpg"
```

















**NOTE**:
In case that `requirements.txt` gives an error during installation (something like this **ERROR: No matching distribution found for torch==2.0.1+cu117**), install it manually as:
```
pip install torch==2.0.1+cu117 --index-url https://download.pytorch.org/whl/cu117
```
Therefore, you will need to install a few more following libraries manually:
```
pip install numpy==1.25.2
pip install torchvision==0.15.2
pip install matplotlib==3.8.0
pip install PyYAML==6.0.1
pip install pytest==7.4.2
```

## Dataset
The cleaned dataset for this task is available [here](https://drive.google.com/drive/folders/1qOVe9A5fX5D5vNGIuDvlIGCT0mq-PbGF).

The dataset consists of two folders:
1. **image-db** - consist of more than 4000 images. 
2. **test-cases** - consists of several query images.

## Guideline
Tree of the project:
```plaintext
project_root/
├── .github/workflows/ 
│   └── main.yaml
├── save_figures/
│   ├── car.jpg
│   ├── cat.jpg
│   ├── face.jpg
│   ├── flower.jpg
│   ├── leopard.jpg
│   └── pizza.jpg
├── source/
│   ├── __init__.py
│   ├── dataset.py
│   ├── vectorizer.py
│   ├── similarity.py
│   ├── search_engine.py
│   ├── test_vectorizer.py
│   ├── test_similarity.py
│   ├── test_search_engin.py
│   ├── my_vector.npy
│   └── image_name.json
├── README.md
├── .gitignore
├── requirements.txt
├── config.yaml
└── main.py
```

- [x] **To run this program**, you should call in the terminal:
```
python main.py
```
- [x] This repository contains configuration file `config.yaml` where you can 
config the roots of your:
1. images dataset
2. query image
3. the name of the collage image with top N similar images

For example: 
- the root of the image dataset is:
```
root_images: "dataset/image-db/"
```
- query image is located in:
```
root_query_image: "dataset/test-cases/cat.jpg"
```
and the name of the collage image is:
```
save_figure_name: "cat.jpg"
```

- [x] The **source** folder contains all the important files for solving the tasks of this assignment.
- `datasset.py` and `vectorizer.py` contain the code to solve the **first task**.
- `similarity.py` contains the code for solving the **second task**.
- `searg_engin.py` and `similarity.py` contain code to solve the **third task**.
- `my_vector.npy` contains vectorized images from image dataset.
- `my_vector.npy` contains vectorized images from image dataset 
- `image_name.json` contains image names from image dataset.

Unit testing is done using the pytest framework. Unit tests are written in `test_vectorizer.py`, `test_similarity.py` and `test_search_engine.py` files. To execute tests, you should run the following operations in a terminal:
```
pytest source/test_vectorizer.py
```
```
pytest source/test_similarity.py
```
```
pytest source/test_search_engin.py
```

- [x] Folder `.github/workflow/` consists the file `main.yaml` with instructions to automaticly execute CI in GitHub Actions. Unfortunately, there is a next error **ERROR: No matching distribution found for torch==2.0.1+cu117** for which a solution has not yet been found. Because of that, it is proposed to execute the test manually using pytest and terminal locally.

## Results
Using query image as input which is uploaded throug `config.yaml` file by providing query image root, in the results we get a collage of the top N most
similar images from the data set sorted by similarity score in descending order. 

As test query images we take:

**Car**:
![Alt](save_figures/car.jpg)
**Cat**:
![Alt](save_figures/cat.jpg)
**Face**:
![Alt](save_figures/face.jpg)
**Flower**:
![Alt](save_figures/flower.jpg)
**Leopard**:
![Alt](save_figures/leopard.jpg)
**Pizza**:
![Alt](save_figures/pizza.jpg)

## Technical Details
<img src="https://render.githubusercontent.com/render/math?math={}">

To solve this assignment the pre-trained convolutional neural networks (CNN) are implemented to transform an image into its vectorized form. As a pre-trained CNN model, the embedding layer of the `ResNet50` pre-trained on the image net is applied to get the feature vector as the image representation. 

To be able to get information on how similar is the new (unseen) image to each one from our processed dataset of images we should implement a similarity function. As the similarity function in this assignment, we implement cosine distance between two feature representations of images, the so-called `cosine similarity function`.

For two given vectors A and B, the cosine similarity function is equal:

<img src="https://render.githubusercontent.com/render/math?math={}">

$$
cos(\Theta) = \frac{A \cdot B }{\lVert A \rVert \cdot \lVert B \rVert}
$$



where ||A|| and ||B|| are Euclidean norm of vectors, and A $\cdot$ B is dot product of two vectors.