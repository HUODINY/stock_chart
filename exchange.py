import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import FinanceDataReader as fdr
import koreanize_matplotlib 
import streamlit as st

# FX 환율, 1995 ~ 현재
usdkrw = fdr.DataReader('USD/KRW', '1995-01-01') # 달러 원화
usdeur = fdr.DataReader('USD/EUR', '1995-01-01') # 달러 유로화
usdcny = fdr.DataReader('USD/CNY', '1995-01-01') # 달러 위엔화

# 스트림릿 템플릿
chart_data = pd.DataFrame(
     np.random.randn(20, 3),
     columns=['usdkrw[[Close]]', 'usdeur[[Close]]', 'usdcny[[Close]]'])

st.line_chart(chart_data)
