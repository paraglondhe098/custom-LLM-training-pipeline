# MiniGPT: Custom Language Model Training Pipeline

MiniGPT is a comprehensive pipeline application designed to train a language model on a custom dataset. It simplifies the process of raw data ingestion, tokenization, model training, and response generation, making it easy to develop and experiment with language models.

---

## Problem
Design a pipeline for training a language model from scratch using a custom dataset. 
Highlight key steps, tools, and challenges. Share Google Drive link to your response.


## Overview

This pipeline takes raw text data as input, tokenizes it, trains a model, and generates text based on the trained model. Its modular design enables users to run individual components or the entire pipeline seamlessly.

---

## Directory Structure

```plaintext
MiniGPT/
├── artifacts/              # Model weights, training history, tokenizer objects etc.
├── data/
│   ├── raw_files/         # Place your raw .txt files here
|   |-- preprocessed/      # Corpus files
├── utils/
│   ├── app.py             # Main application class
│   ├── config.yaml        # Configuration file
│   ├── data_ingestion.py  # Data ingestion pipeline
│   ├── generate.py        # Response generation pipeline
│   ├── main.py            # CLI entry point for the pipeline
│   ├── train.py           # Training pipeline
│   └── train_tokenizer.py # Tokenization pipeline
├── output.txt             # Output results
├── README.md              # Project documentation
├── test.ipynb             # Jupyter notebook for testing
└── requirements.txt       # Python dependencies
```

---

## Prerequisites

1. **Python 3.8 or higher** is required.
2. Install the required dependencies using the following command:

    ```bash
    pip install -r requirements.txt
    ```

---

## Pipeline Steps

### Step 1: Prepare Raw Data

Place your raw text files (in `.txt` format) in the `data/raw_files` directory.

### Step 2: Run the Pipeline

You can execute each pipeline step independently or run the entire pipeline using `main.py`.

#### **1. Data Ingestion**

Reads and preprocesses raw text data for the next steps.

```bash
python main.py ingest
```

#### **2. Text Tokenization**

Tokenizes the preprocessed text using a custom tokenizer.

```bash
python main.py tokenize
```

#### **3. Model Training**

Trains a language model using the tokenized data.

```bash
python main.py train
```

#### **4. Response Generation**

Generates responses based on the trained language model.

```bash
python main.py generate
```

#### **5. Run Complete Pipeline**

To execute all training steps in sequence (ingestion → tokenization → training ):

```bash
python main.py run
```

---

## File Descriptions

### `app.py`

- Central application class managing the pipeline.
- Methods include:
  - `ingest_data()`: Triggers the data ingestion pipeline.
  - `tokenize_text()`: Triggers the tokenization pipeline.
  - `train_model()`: Triggers the training pipeline.
  - `generate_responses()`: Triggers response generation.
  - `run_pipeline()`: Runs all steps sequentially.

### `data_ingestion.py`

Handles the ingestion and preprocessing of raw `.txt` files from `data/raw_files`.

### `train_tokenizer.py`

Defines the tokenization pipeline, trains a custom tokenizer, and saves it for use in subsequent steps.

### `train.py`

Implements the training pipeline, training a language model on the tokenized dataset.

### `generate.py`

Handles response generation using the trained language model.

### `main.py`

Provides a command-line interface (CLI) to run the pipelines.

---

## Example Usage

1. **Prepare your raw text files:**
   Place `.txt` files in the `data/raw_files` directory.

2. **Run the entire pipeline:**
   Use the following command to process data, train the model, and generate responses:

    ```bash
    python main.py run
    ```

3. **View the outputs:**
   - Tokenized data is saved in the output directory of the tokenization pipeline.
   - Trained model weights and history are stored in the `artifacts/` directory.
   - Generated responses are printed or saved based on the configuration.


---

## Future Enhancements

- Support for additional data formats such as `.csv` and `.json`.
- Improvements to the response generation pipeline, including advanced customization options.
- Add visualization tools for monitoring training progress and evaluating model performance.

---
