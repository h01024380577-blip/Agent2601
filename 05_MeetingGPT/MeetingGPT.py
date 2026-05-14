import os, time
from dotenv import load_dotenv

load_dotenv()  #

print(f'✅ {os.path.basename( __file__ )} 실행됨 {time.strftime('%Y-%m-%d %H:%M:%S')}')  # 실행파일명, 현재시간출력
print(f'\tOPENAI_API_KEY={os.getenv("OPENAI_API_KEY")[:20]}...') # OPENAI_API_KEY 필요!
#─────────────────────────────────────────────────────────────────────────────────────────
import streamlit as st
import glob
import subprocess
import math
from pydub import AudioSegment
import openai


from langchain_openai.chat_models.base import ChatOpenAI
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_community.document_loaders.text import TextLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser

from langchain_community.vectorstores.faiss import FAISS
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_openai.embeddings.base import OpenAIEmbeddings
from langchain_classic.storage import LocalFileStore 


# ────────────────────────────────────────
# 🎃 LLM 로직
# ────────────────────────────────────────
llm = ChatOpenAI(
    temperature=0.1,
)

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=800,
    chunk_overlap=100,
)

# podcast.txt 를 embed 해야 한다.
@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file_path):
    cache_dir = LocalFileStore(os.path.join(embedding_dir))  

    loader = TextLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)

    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

    vectorstore = FAISS.from_documents(docs, cached_embeddings)

    retriever = vectorstore.as_retriever()
    return retriever

# ────────────────────────────────────────
# 🍇 file load & cache
# ────────────────────────────────────────
file_dir = os.path.dirname(os.path.realpath(__file__)) # *.py 파일의 '경로'만
# .cache  ← 업로드한 비디오 와 변환한 mp3
# .cache/chunks ← 분할된 mp3 파일들 저장
upload_dir = os.path.join(file_dir, '.cache/chunks')
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)

embedding_dir = os.path.join(file_dir, r'./.cache/embeddings')
if not os.path.exists(embedding_dir):
    os.makedirs(embedding_dir)    



# 🐶수업용 킬스위치: transcript 한번 했으면 중복해서 실행하지 말기
has_transcript = os.path.exists(os.path.join(file_dir, r'.cache/podcast.txt'))

# 오디오 추출함수
# 업로드한 mp4 비디오파일에서 오디오 mp3 추출하여 동일 경로에 저장
@st.cache_resource()
def extract_audio_from_video(video_path):
    if has_transcript: return  # 🐶수업용

    audio_path = video_path.replace('mp4', 'mp3')
    command = ["ffmpeg", "-i", video_path,  "-vn", audio_path, "-y"]
    subprocess.run(command)

# audio_path : 원본 오디오 경로
# chunk_size : minute
# chunks_folder: chunk 들을 저장할 폴더
@st.cache_resource()
def cut_audio_in_chunks(audio_path, chunk_size, chunks_folder):
    if has_transcript: return  # 🐶수업용

    track = AudioSegment.from_mp3(audio_path)
    chunk_len = chunk_size * 60 * 1000
    chunks = math.ceil(len(track) / chunk_len)
    for i in range(chunks):
        start_time = i * chunk_len
        end_time = (i + 1) * chunk_len

        chunk = track[start_time:end_time]

        exp_path = os.path.join(chunks_folder, f"chunk_{i}.mp3")
        chunk.export(exp_path, format="mp3")  


# chunk_folder : 분할된 오디오 파일들 저장 디렉토리
# destination : 녹취록이 들어간 텍스트 파일이 저장될 디렉토리
@st.cache_resource()
def transcribe_chunks(chunk_folder, destination):
    if has_transcript: return  # 🐶수업용
    
    files = glob.glob(os.path.join(chunk_folder, 'chunk*.mp3'))
    files.sort()  # glob() 은 파일명 오름차순 정렬이 보장 안됨!

    for file in files:
        with open(file, 'rb') as audio_file, open(destination, 'a') as text_file: 
            print(file, '녹취록 가져오는 중...', end='')
            transcript = openai.audio.transcriptions.create(
                model='whisper-1',
                file=audio_file,
                language='en',
            )
            text_file.write(transcript.text)            
            print('완료')

# ────────────────────────────────────────
# ⭕ Streamlit 로직
# ────────────────────────────────────────
st.set_page_config(
    page_title="MeetingGPT",
    page_icon="🎤",
)
st.markdown(
    """
# MeetingGPT
            
Welcome to MeetingGPT, upload a video and I will give you a transcript, a summary and a chat bot to ask any questions about it.

Get started by uploading a video file in the sidebar.
"""
)

with st.sidebar:
    video = st.file_uploader(
        label="Video",
        type=["mp4", "avi", "mkv", "mov"],
    )

if video:
    with st.status("Loading video...") as status:
        video_content = video.read()
        video_path = os.path.join(file_dir, rf".cache/{video.name}")
        audio_path = video_path.replace('mp4', 'mp3')
        with open(video_path, 'wb') as f:
            f.write(video_content)

        status.update(label="Extrating audio...")
        # 오디오 변환
        extract_audio_from_video(video_path)

        status.update(label="Cutting audio segments...")
        # mp3 오디오 잘라내기
        chunks_folder = os.path.join(file_dir, r'./.cache/chunks')
        cut_audio_in_chunks(audio_path, 10, chunks_folder)

        status.update(label="Transcripting Audio...")
        transcript_path = video_path.replace("mp4", 'txt')
        transcribe_chunks(chunks_folder, transcript_path)

    # 3개의 tab
    transcript_tab, summary_tab, qa_tab = st.tabs(["Transcript", "Summary", "Q&A"])

    with transcript_tab:
        with open(transcript_path, "r") as file:
            st.write(file.read())    


    with summary_tab:
        start = st.button("Generate summary")

        # ↓ 2개의 chain 을 시작
        #  첫번째 chain : 첫번째 document 를 요약 (summarize)
        #  두번째 chain : 다른 모든 document 를 요약
        #      LLM 에게 '이전의 summary' 와 새 context 를 사용하여 새로운 summary 를 만들게 함 (refine!).
        if start:
            loader = TextLoader(transcript_path)

            docs = loader.load_and_split(text_splitter=splitter)

            # 첫번째 chain
            first_summary_prompt = ChatPromptTemplate.from_template("""
                Write a concise summary of the following:
                "{text}"
                CONCISE SUMMARY:
            """)

            first_summary_chain = first_summary_prompt | llm | StrOutputParser()

            summary = first_summary_chain.invoke({
                "text": docs[0].page_content,   # 첫번째 Document
            })

            # 두번째 chain: 나머지 Document 들을 요약할 chain
            refine_prompt = ChatPromptTemplate.from_template(
                """
                Your job is to produce a final summary.
                We have provided an existing summary up to a certain point: {existing_summary}
                We have the opportunity to refine the existing summary (only if needed) with some more context below.
                ------------
                {context}
                ------------
                Please refine the existing summary using the additional context, if necessary.
                If the additional context does not require any changes to the existing summary, **return the existing summary exactly as it is.**
                Do NOT explain your decision. Just output the final summary.
                """
            )

            refine_chain = refine_prompt | llm | StrOutputParser()

            # 나머지 Document 들 에 대해... 체인호출 (시간 걸림)
            with st.status("Summarizing...") as status:
                for i, doc in enumerate(docs[1:]):
                    status.update(label=f"Processing document {i+1}/{len(docs) - 1}")
                    summary = refine_chain.invoke({
                        "existing_summary": summary,  # 이전 summary 와
                        "context": doc.page_content,  # 다음 Document 를 사용
                    })

                    st.write("🔷 ", summary)  # 확인용  : 중간단계 요약 누적

            st.write("✅ ", summary) # 최종 요약.


    with qa_tab:
        retriever = embed_file(transcript_path)
        docs = retriever.invoke("Do they talk about marcus aurelius?")            
        st.write(docs) # 확인












