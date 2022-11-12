import os
import config
import torch
import engine
import numpy as np
import pandas as pd
import transformers as tns
import torch.nn as nn
from sklearn import model_selection, metrics
from dataset import BertDataset
from model import BertBaseUncased


def run():
    dfx = pd.read_csv(config.TRAIN_FILE).fillna("none").head(100)
    dfx.sentiment = dfx.sentiment.apply(
        lambda x: 1 if x == "positive" else 0
    )

    df_train, df_valid = model_selection.train_test_split(
        dfx,
        test_size=0.1,
        random_state=42,
        stratify=dfx.sentiment.values
    )

    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    train_dataset = BertDataset(
        review=df_train.review,
        target=df_train.sentiment
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=8
    )

    valid_dataset = BertDataset(
        review=df_valid.review,
        target=df_valid.sentiment
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=8
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and config.DEVICE == "cuda"
        else "cpu"
    )
    model = BertBaseUncased().to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            'params': [
                param for name, param in param_optimizer
                if not any(nd in name for nd in no_decay)
            ],
            'weight_decay': 0.0001
        },
        {
            'params': [
                param for name, param in param_optimizer
                if any(nd in name for nd in no_decay)
            ],
            'weight_decay': 0.0
        },
    ]

    num_train_steps = int(df_train.shape[0] / config.TRAIN_BATCH_SIZE * config.EPOCHS)

    optimizer = tns.AdamW(optimizer_parameters, lr=3e-5)
    scheduler = tns.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps
    )

    best_accuracy = 0
    for epoch in range(config.EPOCHS):
        engine.train_fn(
            data_loader=train_data_loader,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            accumulation_steps=1,
            loss_function=engine.loss_fn
        )

        result = engine.eval_fn(
            data_loader=valid_data_loader,
            model=model,
            device=device,
        )

        outputs = np.array(result['probabilities']) > 0.5
        accuracy = metrics.accuracy_score(result['targets'], outputs)

        print(f"Accuracy score = {accuracy}")

        if accuracy > best_accuracy:
            path_to_save = os.path.join("models", "checkpoints", "bert_base_uncased")
            torch.save(model.state_dict(), os.path.join(path_to_save, config.MODEL_PATH))
            best_accuracy = accuracy


if __name__ == "__main__":
    run()







