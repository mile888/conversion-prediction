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
5. number of hyperparameters bayesian optimization calls - (e.g. `n_calls: 30`)
6. model_treshold - (e.g. `model_treshold: 0.4`)
7. `ds_homework_sample.csv` - (e.g. `data: "datasets/ds_homework_sample.csv"`)
8. the nuber of index row from raw dataset - (e.g. `index_row: 1`)

- [x] `main.ipynb` nootebook is used for feature engineering and creating cleaned `train.csv` and `test.csv` datasets

- [x] The **source** folder contains all the important files for solving the tasks of this assignment.
- `utils.py` contains the code for fitting and optimizing classifiers..
- `mrmr.py` contains the code of the so-called minimum redundancy maximum relevance feature engineering approach.
- `train_validate.py` contains pipeline for solving train\validation task. 
- `predict.py` contains a pipeline for solving this task from the raw dataset.

- [x] The **save_models** consists of the saved models and their parameter files.







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