import config
import transformers as tns
import torch.nn as nn


class BertBaseUncased(nn.Module):

    def __init__(self, do: float = 0.3):
        super(BertBaseUncased, self).__init__()
        self.bert = tns.AutoModel.from_pretrained(config.BERT_PATH)
        self.bert_do = nn.Dropout(do)
        self.out = nn.Linear(512, 1)

    def forward(self, ids, mask, token_type_ids):
        output = self.bert(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        bo = self.bert_do(output.pooler_output)
        bo = self.out(bo)
        return bo
