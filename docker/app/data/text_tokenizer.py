import os
import torch
from data.data import set_data
from data.tokenizer_utils import normalize_text
from data.tokenizer_utils import train_and_load_sp_models
from utils.config import CONFIG # Import CONFIG dictionary
from tqdm import tqdm
import sentencepiece as spm
import multiprocessing
