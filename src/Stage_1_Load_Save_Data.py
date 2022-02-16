import os
import yaml
import pandas as pd
import argparse

def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def get_data(config_path):
    config_path = read_params(config_path)
    data_path = config_path["data_source"]["s3_source"]
    df = pd.read_csv(data_path, sep=";", encoding='utf-8')
    return df

def load_and_save(config_path):
    config = read_params(config_path)
    df = get_data(config_path)
    new_cols = [col.replace(" ", "_") for col in df.columns]
    raw_data_path = config["load_data"]["raw_dataset_csv"]
    print(df)
    df.to_csv(raw_data_path, sep=",", index=False, header=new_cols)

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="/home/shubham/dvc57/paths.yaml")
    parsed_args = args.parse_args()
    load_and_save(config_path=parsed_args.config)