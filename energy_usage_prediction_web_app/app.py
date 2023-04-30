import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import statsmodels.tsa.seasonal as sea
from dateutil.parser import parse
from tensorflow import keras

model = keras.models.load_model("model")
print(model)

arr = np.random.normal(1, 1, size=100)
fig, ax = plt.subplots()
ax.hist(arr, bins=20)

# st.pyplot(fig)
fig
