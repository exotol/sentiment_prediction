import config
import numpy as np
import torch
from typing import Dict, Any, Optional


class BertDataset:

    def __init__(self, review: np.array, target: Optional[np.array] = None):
        self.review = review
        self.target = target
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN

    def __len__(self) -> int:
        return len(self.review)

    def __getitem__(self, index) -> Dict[str, Any]:
        review = str(self.review)
        review = " ".join(review.split())

        inputs = self.tokenizer.encode_plus(
            review,
            None,
            add_special_tokens=True,
            max_length=self.max_len
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']

        padding_len = self.max_len - len(ids)
        ids = ids + ([0] * padding_len)
        mask = mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)
        if self.target is not None:
            return dict(
                ids=torch.tensor(ids, dtype=torch.long),
                mask=torch.tensor(mask, dtype=torch.long),
                token_type_ids=torch.tensor(token_type_ids, dtype=torch.long),
                target=torch.tensor(self.target[index], dtype=torch.float)
            )
        return dict(
            ids=torch.Tensor(ids, dtype=torch.long),
            mask=torch.Tensor(mask, dtype=torch.long),
            token_type_ids=torch.Tensor(token_type_ids, dtype=torch.long),
        )