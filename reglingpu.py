# -*- coding: utf-8 -*-
"""
Poprawiona implementacja regresji liniowej w PyTorch dla danych FIFA
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import time
import matplotlib.pyplot as plt
print(f"Czy CUDA jest dostępne: {torch.cuda.is_available()}")
print(f"Nazwa karty graficznej: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Brak GPU'}")
