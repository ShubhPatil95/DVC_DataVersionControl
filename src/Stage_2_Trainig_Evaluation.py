# load the train and test
# train algo
# save the metrices, params
import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import argparse
import joblib
import json
import yaml

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def read_params(config_path):
    with open(config_path) as yaml_file:
      config=yaml.safe_load(yaml_file)
    return config

def train_and_evaluate(config_path):
    config = read_params(config_path)
    config_path=read_params(config["path"]["path"])
    random_state = config["base"]["random_state"]
    model_dir = config_path["model_dir"]["model_dir"]
    raw_data_path = config_path["load_data"]["raw_dataset_csv"]
    split_ratio = config["split_data"]["test_size"]

    alpha = config["estimators"]["ElasticNet"]["params"]["alpha"]
    l1_ratio = config["estimators"]["ElasticNet"]["params"]["l1_ratio"]

    target = [config["base"]["target_col"]]

    df = pd.read_csv(raw_data_path, sep=",")
    train, test = train_test_split(
        df, 
        test_size=split_ratio, 
        random_state=random_state
        )
    
    train_y = train[target]
    test_y = test[target]

    train_x = train.drop(target, axis=1)
    test_x = test.drop(target, axis=1)

    lr = ElasticNet(
        alpha=alpha, 
        l1_ratio=l1_ratio, 
        random_state=random_state)
    lr.fit(train_x, train_y)

    predicted_qualities = lr.predict(test_x)
    
    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

#####################################################
    scores_file = config_path["reports"]["scores"]
    
    with open(scores_file, "w") as f:
        scores = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }
        json.dump(scores, f, indent=4)
        
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")

    joblib.dump(lr, model_path)        
 
if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="/home/shubham/dvc57/params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)
