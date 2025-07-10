# 실습 2 : Gemini API를 활용한 간단한 챗봇
import streamlit as st
from dotenv import load_dotenv

load_dotenv("../../.env")

import os
from google import genai
from google.genai import types
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

system_prompt = """
사용자의 질문에 대해 친절하게 답변하세요.
답변은 2문장을 넘지 않습니다.
"""

# 페이지 제목
st.title("실습 2: Gemini API를 사용하는 챗봇")

# 진행된 대화내용을 저장하도록 session_state 활용
# 아직 저장된 내용이 없다면, 초기화
if "messages" not in st.session_state:
  st.session_state["messages"] = []

# 저장된 내용이 있다면, 새로운 대화를 추가전에 화면 출력
for message in st.session_state.messages:
  with st.chat_message(message["role"]):
    st.markdown(message["content"])

# 새로운 사용자 입력을 처리한다.
if prompt := st.chat_input("무엇이든 물어보랏"):
  with st.chat_message("user"):
    st.markdown(prompt)
  
  # 새로 들어온 입력을 기존 대화내용에 추가한다.
  st.session_state.messages.append(
    {"role": "user", "content": prompt})
  
  # 챗봇은 Gemini API를 사용하여 답변을 생성한다.
  bot_response = client.models.generate_content(
    model="gemini-2.5-flash",
    config=types.GenerateContentConfig(
        system_instruction=system_prompt),
    contents=prompt
    )

  with st.chat_message("assistant"):
    st.markdown(bot_response.text)

  # 새로 구성한 챗봇의 대답을 대화내용에 추가한다.
  st.session_state.messages.append(
    {"role": "assistant", "content":bot_response.text})