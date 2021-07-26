import pydeck as pdk
import datetime
import bar_chart_race as bcr
import math
import altair as alt
from altair import Chart, X, Y, Axis, SortField, OpacityValue
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import numpy as np
import time
import streamlit as st


@st.cache(suppress_st_warning=True) #adding cache
def expensive_computation(a, b):
    st.write("Cache miss: expensive computation(", a, ",", b, ") ran")
    time.sleep(2) #This makes the function take two seconds to run
    return {"output": a * b}

a = 2
b = 21
#b = st.slider("Pick a number", 0, 10)
res = expensive_computation(a, b)

st.write("Result", res)

res["output"] = "result was manually mutated"

st.write("Mutated reuslt:", res)