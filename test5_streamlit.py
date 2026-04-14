import streamlit as st
import os, time

print(f'✅ {os.path.basename( __file__ )} 실행됨 {time.strftime('%Y-%m-%d %H:%M:%S')}')  # 실행파일명, 현재시간출력

st.title('hello streamlit')
st.text(time.strftime('%Y-%m-%d %H:%M:%S'))

st.slider('온도', min_value=-10.0, max_value=37.5)

# 실행
# streamlit run 파일명