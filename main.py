
from source.train_validate import TrainModel
from source.predict import PredictConversions

import yaml

# Load config file
with open('config.yaml', 'r') as config_file:
    config_data = yaml.safe_load(config_file)


# Assign values from the config file to the variables
dataset_root_train = config_data['dataset_root_train']
dataset_root_test = config_data['dataset_root_test']
model = config_data['model']
k_best_features = config_data['k_best_features']
n_calls = config_data['n_calls']
model_treshold = config_data['model_treshold']
data = config_data['data']
index_row = config_data['index_row']
mode = config_data['mode']



if __name__ == "__main__":
    # Create dataset instance and call get_image method
    train = TrainModel(dataset_root_train = dataset_root_train, 
                       dataset_root_test = dataset_root_test, 
                       model = model, 
                       k_best_features = k_best_features,
                       model_treshold = model_treshold,
                       n_calls = n_calls)
    
    predict = PredictConversions(data = data,
                                 model = model,
                                 index_row = index_row)
                    
    if mode == 'train':
        cp_prob = train.train_model()
        print("Conversions probability:", cp_prob)
    elif mode == 'predict':
        cp_prob = predict.predict()
        print("Conversions probability:", cp_prob[0])


