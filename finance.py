import pandas as pd
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
