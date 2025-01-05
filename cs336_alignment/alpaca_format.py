from typing import Any
import re, os
from glob import glob
import pandas as pd
from utils.io import load_jsonl

ALPACA_PATH = "/scratch/users/arjo/spring2024-assignment5-alignment/data/alpaca_eval/alpaca_eval"

class alpaca:
    def load_data():
        return load_jsonl(ALPACA_PATH)

if __name__=="__main__":
    alpaca.load_data()
