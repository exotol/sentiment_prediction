import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch import device as device_machine
from tqdm import tqdm
from typing import Callable, Any, Union
from model import BertBaseUncased


def loss_fn(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))


def train_fn(
        data_loader: DataLoader,
        model: BertBaseUncased,
        optimizer: Optimizer,
        scheduler: Any,
        device: device_machine,
        accumulation_steps: int,
        loss_function: Callable
) -> None:
    model.train()

    for batch_index, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        ids = data["ids"].to(device)
        token_type_ids = data["token_type_ids"].to(device)
        mask = data["mask"].to(device)
        targets = data["target"].to(device)

        optimizer.zero_grad()
        outputs = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )

        loss = loss_function(outputs, targets)
        loss.backward()

        if (batch_index + 1) % accumulation_steps == 0:
            optimizer.step()
            scheduler.step()


def eval_fn(
        data_loader: DataLoader,
        model: BertBaseUncased,
        device: device_machine,
):
    model.eval()

    fin_targets = []
    fin_outputs = []

    with torch.no_grad():
        for batch_index, data in tqdm(enumerate(data_loader), total=len(data_loader)):
            ids = data["ids"].to(device)
            token_type_ids = data["token_type_ids"].to(device)
            mask = data["mask"].to(device)
            targets = data["targets"].to(device)

            outputs = model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )
            probabilities = torch.sigmoid(outputs)

            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(probabilities.cpu().detach().numpy().tolist())
    return dict(
        probabilities=fin_outputs,
        targets=fin_targets
    )

