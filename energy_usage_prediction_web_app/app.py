import datetime
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd
import statsmodels.tsa.seasonal as sea
import streamlit as st
from dateutil.parser import parse
from tensorflow import keras

# I hate the ~~antichrist~~ seaborn docs regarding matplotlib integration
st.set_option("deprecation.showPyplotGlobalUse", False)

st.write("# Energy usage analysis & prediction toolkit")

model = keras.models.load_model("model")


def decompose(df):
    # I'm too lazy to figure out how to render multiple
    # plots at once, so I'm only using a single decompose model

    # Multiplicative Decomposition
    # mul = sea.seasonal_decompose(df['Usage'], period=24, model='multiplicative')

    # Additive Decomposition
    add = sea.seasonal_decompose(df["Usage"], period=24, model="additive")

    # Plot
    plt.rcParams.update({"figure.figsize": (16, 12)})
    # mul.plot().suptitle('Multiplicative Decomposition', fontsize=16)
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    add.plot().suptitle("Additive Decomposition", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    st.pyplot()


def przewidywania_gartnera(model, real_y, predict, in_size):
    y = list(real_y)[:-predict]
    for i in range(predict):
        inp = np.array([y[-in_size:]])
        value = model.predict(inp, verbose=False).flatten()[0]
        y.append(value)
    x = list(range(len(real_y)))
    plt.plot(x[-predict * 2 :], y[-predict * 2 :], label="Predykcja")
    plt.plot(x[-predict * 2 :], real_y[-predict * 2 :], label="Prawdziwe wartości")
    plt.scatter(
        len(x) - predict - 1, real_y[len(x) - predict - 1], c="r", label="Start"
    )
    plt.legend()
    st.pyplot()


def parse_uploaded_file(file):
    bytes_data = file.getvalue()
    df = pandas.read_csv(BytesIO(bytes_data))

    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.sort_values("Datetime")
    df = df.set_index("Datetime")

    mw_usage_column = df.columns[0]
    df = df.rename({mw_usage_column: "Usage"}, axis="columns")
    df["Usage"] = df["Usage"] / max(df["Usage"])

    return df


uploaded_file = st.file_uploader("Choose an energy measurements file")
analysis_tab, prediction_tab = st.tabs(["Analysis", "Prediction"])

if uploaded_file is not None:
    df = parse_uploaded_file(uploaded_file)

    with analysis_tab:
        decomposition_start_date = st.date_input(
            "Date to perform decomposition from", datetime.date(2006, 1, 1)
        )
        decomposition_end_date = decomposition_start_date + datetime.timedelta(days=60)

        decompose(df[decomposition_start_date:decomposition_end_date])

    with prediction_tab:
        przewidywania_gartnera(model, df["Usage"], 101, 10 * 24)
