from train import training_pipeline
from data_ingesstion import data_ingestion_pipeline
from train_tokenizer import tokenization_pipeline
from generate import generation_pipeline


class App:
    def __init__(self):
        pass

    def run_pipeline(self):
        self.ingest_data()
        self.tokenize_text()
        self.train_model()

    @staticmethod
    def generate_responses():
        generation_pipeline()

    @staticmethod
    def ingest_data():
        data_ingestion_pipeline()

    @staticmethod
    def tokenize_text():
        tokenization_pipeline()

    @staticmethod
    def train_model():
        training_pipeline()





