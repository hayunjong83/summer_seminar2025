# 실습 1 : 환각현상을 제어하기 위한 RAG
import streamlit as st
from dotenv import load_dotenv

load_dotenv("../../.env")

import os
from google import genai
from google.genai import types
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

system_prompt = """
사용자의 질문에 대해 친절하게 답변하세요.
200자 내외로 답변해주세요.
"""

# 페이지 제목
st.title("실습 1: RAG를 이용한 챗봇")

import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

@st.cache_resource
def init_rag():
  # 문서 (간단한 예제 사용 — 원하는 텍스트 파일로 교체 가능)
  documents = [
      "덕성여자대학교는 서울특별시 도봉구에 위치한 4년제 사립 종합 여자대학교로, 여성 독립운동가 차미리사 선생이 최초로 설립한 여학교에서 시작되었습니다. 이 대학의 전신은 1920년 설립된 근화학원으로, 이후 근화여자실업학교를 거쳐 1938년 덕성여자실업학교로 개명되었습니다. 덕성여대는 ‘덕성을 갖춘 창의적 지식인 육성’이라는 교육 이념을 바탕으로 운영되며, 특히 유아교육 분야에서 강점을 보입니다. 2020년에는 창학 100주년을 맞이하였습니다.",
      "덕성여자대학교의 창학 이념은 '자생(自生), 자립(自立), 자각(自覺)'으로, 이는 학교의 설립자인 차미리사 선생이 근화여학교를 설립할 당시 정한 것입니다. 이 이념은 ‘살되, 네 생명을 살아라. 생각하되, 네 생각으로 하여라. 알되, 네가 깨달아 알아라.’라는 문구로 표현되며, 여성의 자립과 깨달음을 강조하는 교육 철학을 담고 있습니다.",
      "덕성여자대학교는 두 개의 캠퍼스를 운영하고 있습니다.(1) 쌍문동캠퍼스: 서울 경전철 우이신설선 4.19민주묘지역(덕성여대역) 근처에 위치하며, 재학생들이 주로 이용하는 본 캠퍼스입니다.(2) 종로캠퍼스: 서울 지하철 3호선 안국역 4번 출구 부근에 위치하여, 주로 교육 및 연구 활동을 위한 공간으로 활용됩니다.",
      "덕성여자대학교의 초대 학장은 송금선이며, 1950년 5월 17일부터 1970년 7월 14일까지 재임하였습니다. 초대 총장은 김종서이며, 1988년 3월 1일부터 1989년 8월 20일까지 재임하였습니다."
  ]
  ids = [f"doc_{i}" for i in range(len(documents))]

  # 임베딩 모델 로드
  model = SentenceTransformer("all-MiniLM-L6-v2")
  embed_f = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

  # ChromaDB 인메모리 DB 초기화
  db = chromadb.Client()
  collection = db.create_collection(name="rag_docs", embedding_function=embed_f)

  # 문서 삽입
  collection.add(documents=documents, ids=ids)

  return collection

rag_collection = init_rag()

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
  
  results = rag_collection.query(query_texts=[prompt], n_results=2)
  retrieved_chunks = results["documents"][0]
  context = "\n".join(retrieved_chunks)

  # Gemini에 질문 + 문서 context 포함하여 전송
  combined_prompt = f"""질문: {prompt}    
  (아래 문서를 참고하여 답변하세요)
  ---
  {context}
  ---"""

  # 맥락을 유지하면서 Gemini API를 사용하며 답변을 생성하게 한다.
  bot_response_w_context = st.session_state.chat_session.send_message(combined_prompt)
  # bot_response_w_context = st.session_state.chat_session.send_message(prompt)

  with st.chat_message("assistant"):
    st.markdown(bot_response_w_context.text)

  # 새로 구성한 챗봇의 대답을 대화내용에 추가한다.
  st.session_state.messages.append(
    {"role": "assistant", "content":bot_response_w_context.text})
  