import os
import time

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

print(f"✅ {os.path.basename(__file__)} 실행됨 {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Streamlit Cloud에서는 st.secrets, 로컬에서는 .env 둘 다 지원
api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")

if not api_key:
    st.error(
        "⚠️ OPENAI_API_KEY가 설정되지 않았습니다.\n\n"
        "**로컬 실행**: 프로젝트 루트에 `.env` 파일을 만들고 `OPENAI_API_KEY=sk-...` 추가\n\n"
        "**Streamlit Cloud**: Manage app → Settings → Secrets에 "
        "`OPENAI_API_KEY = \"sk-...\"` 추가"
    )
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key


from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_openai.chat_models.base import ChatOpenAI
from langchain_core.runnables.base import RunnableLambda
from langchain_core.runnables.passthrough import RunnablePassthrough

# 가벼운 file-format별 loader 사용 (UnstructuredFileLoader 대신)
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)

from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_openai.embeddings.base import OpenAIEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.callbacks.base import BaseCallbackHandler


# ────────────────────────────────────────
# 🎃 LLM 로직
# ────────────────────────────────────────

class ChatCallbackHandler(BaseCallbackHandler):
    def on_llm_start(self, *args, **kwargs):
        self.message = ""
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[ChatCallbackHandler()],
)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


prompt = ChatPromptTemplate.from_messages([
    ("system", """
            Answer the question using ONLY the following context.
            If you don't know the answer just say you don't know. DON'T make anything up.

            Context: {context}
    """),
    ("human", "{question}"),
])


# ────────────────────────────────────────
# 🍇 file load & cache
# ────────────────────────────────────────
upload_dir = r'./.cache/files'
embedding_dir = r'./.cache/embeddings'
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)
if not os.path.exists(embedding_dir):
    os.makedirs(embedding_dir)


def get_loader(file_path: str):
    """파일 확장자에 따라 적절한 loader 반환"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return PyPDFLoader(file_path)
    elif ext == ".docx":
        return Docx2txtLoader(file_path)
    elif ext == ".txt":
        return TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"지원하지 않는 파일 형식: {ext}")


@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = os.path.join(upload_dir, file.name)

    with open(file_path, "wb") as f:
        f.write(file_content)

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )

    loader = get_loader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)

    cahce_dir = LocalFileStore(os.path.join(embedding_dir, file.name))
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cahce_dir)

    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def save_message(message, role):
    st.session_state['messages'].append({"message": message, 'role': role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state['messages']:
        send_message(
            message['message'],
            message['role'],
            save=False,
        )


# ────────────────────────────────────────
# ⭕ Streamlit 로직
# ────────────────────────────────────────
st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)

st.title("Document GPT")

st.markdown(
    """
안녕하세요!

이 챗봇을 사용해서 여러분의 파일들에 대해 AI에게 물어보세요!
"""
)


file = st.file_uploader(
    label="Upload a .txt .pdf or .docx file",
    type=["pdf", 'txt', 'docx'],
)

if file:
    retriever = embed_file(file)

    send_message("준비되엇습니다. 질문해보세요", "ai", save=False)
    paint_history()
    message = st.chat_input("업로드한 file 에 대해 질문을 남겨보세요...")
    if message:
        send_message(message, "human")

        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )

        with st.chat_message("ai"):
            chain.invoke(message)


else:
    st.session_state['messages'] = []
