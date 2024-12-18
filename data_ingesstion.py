import yaml
from utils.preprocessing import TextProcessor


def load_config(path):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def data_ingestion_pipeline():
    config = load_config("config.yaml")
    processor = TextProcessor(config['data']['raw_data_path'])
    processor.create_corpus(config['data']['processed_data_path'])


if __name__ == '__main__':
    data_ingestion_pipeline()
