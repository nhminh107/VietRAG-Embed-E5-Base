import pandas as pd
from database.sql_manager import SQL_Manager, FinanceModel
from tokenizers import Tokenizer
from sentence_transformers import SentencesDataset, SentenceTransformer

MAX_TOKEN = 512
tokenizer = Tokenizer(model="Xenova/multilingual-e5-base")

