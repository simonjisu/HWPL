from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

from transformers.data.processors.squad import *
repo_path = Path().absolute()
data_path = repo_path.parent / "data" / "AIhub" / "QA"
ckpt_path = repo_path.parent / "ckpt"

filename = "val_cache_all"
processed_file = ckpt_path / filename
cache = torch.load(processed_file)
dataset, examples, features = cache["dataset"], cache["examples"], cache["features"]

for f in features[:10]:
    print(f.unique_id)