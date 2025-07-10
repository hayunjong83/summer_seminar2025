# 실습 2-2 : Gemini API를 활용한 멀티턴 (스트리밍) 챗봇
import streamlit as st
from dotenv import load_dotenv

load_dotenv("../../.env")

import os
from google import genai
from google.genai import types
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

system_prompt = """
사용자의 질문에 대해 친절하게 답변하세요.
답변은 500자 내외로 해줘.
"""

# 페이지 제목
st.title("실습 2-2: 스트리밍 챗봇")

# 진행된 대화내용을 저장하도록 session_state 활용
# 아직 저장된 내용이 없다면, 초기화
if "messages" not in st.session_state:
  st.session_state["messages"] = []

# 맥락(context)을 기억하는 멀티턴 대화가 가능해야 한다.
# 이를 가능하게 하는 chats 메소드를 사용한다.
# 단, 답변 생성 시마다 새로운 모델 세션을 선언하지 않고
# 처음에 생성된 모델 세션을 반복해서 사용하도록 한다.
if "chat_session" not in st.session_state:
  st.session_state["chat_session"] = client.chats.create(
    model="gemini-2.5-flash",
    config=types.GenerateContentConfig(
        system_instruction=system_prompt)
    )

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
  
  full_text = [""]
  def stream_data():
    for chunk in st.session_state.chat_session.send_message_stream(prompt):
      full_text[0] += chunk.text
      yield chunk.text

  with st.chat_message("assistant"):
    st.write_stream(stream_data())
  
  # 새로 구성한 챗봇의 대답을 대화내용에 추가한다.
  st.session_state.messages.append(
    {"role": "assistant", "content": full_text[0]})
  