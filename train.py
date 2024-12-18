import os

import torch
import yaml
from utils.preprocessing import TokenizeText, TextDataset
from utils.model import GPTLanguageModel
from utils.torchtrainer.trainer import Trainer
from torch.utils.data import DataLoader
from utils.loss_fn import ModifiedCrossEntropyLoss
from utils.torchtrainer.callbacks import IntraEpochReport0, EarlyStopping0
from utils.state_management import save_state, load_recent
import matplotlib.pyplot as plt


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

batch_size = config['training']['batch_size']
epochs = config['training']['epochs']
optimizer_name = config['training']['optimizer']
learning_rate = config['training']['learning_rate']
load_recent_run = config['training']['load_recent']
split_ratio = config['training']['split_ratio']

# IntraEpochReport
reports_per_epoch = config['training']['reports_per_epoch']

# EarlyStopping
early_stopping = config['training']['early_stopping']
patience = config['training']['patience']
state_save_path = config['training']['state_save_path']
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")


def init_data():
    tokenizer_obj = TokenizeText(corpus_file, tokenizer_save_path, mode="load")
    train_ds = TextDataset(tokenizer_obj, corpus_file, block_size, random=False, split="train", split_ratio=split_ratio)
    val_ds = TextDataset(tokenizer_obj, corpus_file, block_size, random=False, split="val", split_ratio=split_ratio)

    train_dl = DataLoader(dataset=train_ds, batch_size=batch_size)
    val_dl = DataLoader(dataset=val_ds, batch_size=batch_size)
    return tokenizer_obj, train_dl, val_dl


optims = {'AdamW': torch.optim.AdamW, 'Adam': torch.optim.Adam}


def init_model():
    tokenizer_obj, train_dl, val_dl = init_data()

    model = GPTLanguageModel(
        vocab_size=tokenizer_obj.tokenizer.vocab_size,
        n_embd=n_embd,
        block_size=block_size,
        n_head=n_head,
        n_layer=n_layer,
        dropout=dropout
    ).to(device)
    if load_recent_run:
        load_recent(model=model, device=device)
    print(f'Initialized model with {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters')
    return train_dl, val_dl, model


def save_plot(history, filepath):
    # there is some problem with matplotlib and seaborn
    pass
    # plt.plot(history['loss'].va, label='Training Loss')
    # plt.plot(list(history['val_loss']), label='Validation Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.title('Loss vs. Epochs')
    # plt.legend()
    # plt.grid(True)
    # save_path = os.path.join(filepath, "plots.png")
    # plt.savefig(save_path, dpi=300, bbox_inches='tight')  # Save as PNG with high resolution
    # plt.show()


def training_pipeline():
    callbacks = []
    train_dl, val_dl, model = init_model()
    loss_fn = ModifiedCrossEntropyLoss().to(device)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    optimizer = optims[optimizer_name](model.parameters(), lr=learning_rate)
    ier = IntraEpochReport0(reports_per_epoch)
    callbacks.append(ier)
    if early_stopping:
        es = EarlyStopping0(basis="val_loss", patience=patience)
        callbacks.append(es)
    trainer = Trainer(model,
                      epochs=epochs,
                      criterion=loss_fn,
                      input_shape=(32, 128),
                      output_shape=(32, 128, 11057),
                      optimizer=optimizer,
                      callbacks=callbacks,
                      display_time_elapsed=True,
                      metrics=[],
                      device=device)
    history = trainer.fit(train_dl, val_dl)
    run_dir = save_state(trainer)
    save_plot(history, run_dir)


if __name__ == '__main__':
    training_pipeline()
