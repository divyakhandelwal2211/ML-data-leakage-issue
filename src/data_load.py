# src/data_loader.py

import pandas as pd

def load_data(path):
    """
    Load dataset from given path
    """
    df = pd.read_csv(path)
    return df