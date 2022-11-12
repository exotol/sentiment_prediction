import os
from dotenv import load_dotenv
import transformers as tns


load_dotenv()

PROJECT_PATH = os.environ["PROJECT_PATH"]

MAX_LEN = 512
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 2
EPOCHS = 10
ACCUMULATION = 2

# BERT_PATH = os.path.join(PROJECT_PATH, "models/bert_base_uncased")
BERT_PATH = "nlpaueb/legal-bert-small-uncased"
MODEL_PATH = "pytorch_model.bin"

TRAIN_FILE = os.path.join(PROJECT_PATH, "data/input/IMDB Dataset.csv")
TOKENIZER = tns.AutoTokenizer.from_pretrained(
    BERT_PATH
)
DEVICE = "cuda"
