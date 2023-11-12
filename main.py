__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
# from dotenv import load_dotenv
# load_dotenv()
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA
import streamlit as st
import os
import tempfile
from streamlit_extras.buy_me_a_coffee import button

button(username="bigsnail", floating=True, width=221)

st.title("ChatPDF! 섹스보다 재밌다!")
st.write("---")

#OpenAi API키 입력 받기
user_openai_api_key = st.text_input('OPEN_AI_API_KEY', type="password")

uploaded_file = st.file_uploader("파일 업로드", type=['pdf'])
st.write("---")

def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages


#업로드 되면 동장하는 코드
if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)




    #스플리터
    text_splitter = CharacterTextSplitter(
        separator = "\n\n",
        chunk_size = 300,
        chunk_overlap  = 20,
        length_function = len,
        is_separator_regex = False
    )

    texts = text_splitter.split_documents(pages)

    #임베드
    embeddings_model = OpenAIEmbeddings(openai_api_key=user_openai_api_key)

    #크로마로 불러오기
    db = Chroma.from_documents(texts, embeddings_model)

    #헤더
    st.header("PDF를 기반으로 질문해주세요")

    #질문 입력
    question = st.text_input('질문을 입력해주세요')
    
    #버튼 생성
    if st.button('시작'):

        #질문
        llm = ChatOpenAI(model_name = "gpt-3.5-turbo-16k", temperature=0, openai_api_key=user_openai_api_key)
        qa_chain = RetrievalQA.from_chain_type(llm,retriever=db.as_retriever())
        result = qa_chain({"query": question})

        st.write(result)

