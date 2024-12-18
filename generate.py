import yaml
from utils.preprocessing import TokenizeText
import torch
from utils.model import GPTLanguageModel
from utils.state_management import load_recent


def load_config(path):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config


config = load_config("config.yaml")
corpus_file = config['data']['processed_data_path']
tokenizer_save_path = config['data']['tokenizer_save_path']
block_size = config['model']['block_size']
n_embd = config['model']['n_embd']
n_head = config['model']['n_head']
n_layer = config['model']['n_layer']
dropout = config['model']['dropout']
chunksize = config['generation']['chunksize']
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")


def load_recent_model(tokenizer):
    model = GPTLanguageModel(
        vocab_size=tokenizer.tokenizer.vocab_size,
        n_embd=n_embd,
        block_size=block_size,
        n_head=n_head,
        n_layer=n_layer,
        dropout=dropout
    ).to(device)
    load_recent(model=model, device=device)
    return model


def generation_pipeline():
    config = load_config("config.yaml")
    tokenizer_obj = TokenizeText(corpus_file, tokenizer_save_path, mode='load')
    output__file = config['generation']['file']
    context = input("Enter some context: ")
    tokenized_context = tokenizer_obj.tokenize(context)['input_ids'].to(device=device)

    model = load_recent_model(tokenizer_obj)
    generated_text = tokenizer_obj.untokenize(model.generate(tokenized_context, max_new_tokens=300)[0].tolist())
    print("Generated Text:")

    words = generated_text.split()  # Split the sentence into words
    chunk_size = chunksize
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    for chunk in chunks:
        print("".join(chunk))
    # print(generated_text)
    with open(output__file, "w") as file:
        file.write(generated_text)


if __name__ == '__main__':
    generation_pipeline()
