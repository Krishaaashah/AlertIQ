import pandas as pd
import numpy as np
from .config import RANDOM_SEED

np.random.seed(RANDOM_SEED)

def load_and_clean(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    
    # (Moved ID drop to dataset_builder.py so alert rules can group by nameOrig)
    
    # One-hot encode transaction type
    df = pd.get_dummies(df, columns=['type'], drop_first=True)
    
    return df
