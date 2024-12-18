import yaml
from utils.preprocessing import TokenizeText


def load_config(path):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config


config = load_config("config.yaml")
corpus_file = config['data']['processed_data_path']
tokenizer_save_path = config['data']['tokenizer_save_path']


def tokenization_pipeline():
    tokenizer_obj = TokenizeText(corpus_file, tokenizer_save_path)


if __name__ == '__main__':
    tokenization_pipeline()
