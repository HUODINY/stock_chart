import pandas as pd
import numpy as np
import streamlit as st
import FinanceDataReader as fdr
import matplotlib.pyplot as plt
import koreanize_matplotlib

Stockcode = pd.read_csv('data/Stockcode.csv')
Stockcode.set_index('Name', inplace=True)
Name = st.text_input('Code Name')
Code_name_list = Stockcode.index.tolist()

if Name in Code_name_list:
    code_num = Stockcode.at[Name, 'Symbol']
    df = fdr.DataReader(code_num)
    df = df.rename(columns={'Open':'시가', 'High':'고가','Low':'저가', 'Close':'종가', 'Volume':'거래량', 'Change':'전일대비'})
    col1, col2, col3 = st.columns(3)
    col1.metric("현재 주식가격",format(df['종가'].tail(1)[0], ',')+'원', "%d원" %(df['종가'].diff().tail(1)[0]))
    col2.metric("현재 거래량", format(df['거래량'].tail(1)[0], ','),"%.2f%%" %(df['거래량'].pct_change().tail(1)[0] * 100))
    col3.metric("전일 대비 가격", round(df['전일대비'].tail(1)[0], 4), "%.2f%%" %(df['전일대비'].tail(1)[0] * 100))
    fig = plt.figure(facecolor='white', figsize=(20, 10))
    plt.plot(df['종가'])
    plt.title(Name)
    st.pyplot(fig)
elif Name not in Code_name_list:
    st.text('검색하신 주식 종목이 없습니다. 정확하게 입력해주세요.')

    
    # FX 환율, 1995 ~ 현재
usdkrw = fdr.DataReader('USD/KRW', '1995-01-01') # 달러 원화
usdeur = fdr.DataReader('USD/EUR', '1995-01-01') # 달러 유로화
usdcny = fdr.DataReader('USD/CNY', '1995-01-01') # 달러 위엔화

# 스트림릿 템플릿
chart_data = pd.DataFrame(
     np.random.randn(20, 3),
     columns=['usdkrw[[Close]]', 'usdeur[[Close]]', 'usdcny[[Close]]'])

st.line_chart(chart_data)

"""유선님 """
fig = px.line(df, y='종가', title='{}의 종가(close) Time Series'.format(Name))

fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=3, label="3m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(step="all")
        ])
    )
)
fig.show()
