from typing import Any
import re, os
from glob import glob
import pandas as pd
from utils.io import load_jsonl

SST_PATH = "/scratch/users/arjo/spring2024-assignment5-alignment/data/simple_safety_tests/simple_safety_tests.csv"

class sst:
    def load_data(set_type):
        return pd.read_csv(SST_PATH)

if __name__=="__main__":
    sst.load_data()
