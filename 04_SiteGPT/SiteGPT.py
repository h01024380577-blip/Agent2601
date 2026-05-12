import os, time
from dotenv import load_dotenv

load_dotenv()

print(f'✅ {os.path.basename( __file__ )} 실행됨 {time.strftime('%Y-%m-%d %H:%M:%S')}')  # 실행파일명, 현재시간출력
print(f'\tOPENAI_API_KEY={os.getenv("OPENAI_API_KEY")[:20]}...') # OPENAI_API_KEY 필요!
#─────────────────────────────────────────────────────────────────────────────────────────
import streamlit as st

from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai.embeddings.base import OpenAIEmbeddings
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_core.runnables.base import RunnableLambda
from langchain_openai.chat_models.base import ChatOpenAI
from langchain_core.prompts.chat import ChatPromptTemplate

from langchain_community.document_loaders.sitemap import SitemapLoader

# ────────────────────────────────────────
# 🎃 LLM 로직
# ────────────────────────────────────────
llm = ChatOpenAI(
    temperature=0.1,
)

answers_prompt = ChatPromptTemplate.from_template("""
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                 
    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}
                                                 
    Examples:
                                                 
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                 
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                 
    Your turn!

    Question: {question}
""")

def get_answers(inputs):
    docs = inputs['docs']
    question = inputs['question']
    
    # 각 Document 를 처리할 chain
    answers_chain = answers_prompt | llm

    # 다음과 같은 형식으로 리턴해볼거다
    # {
    #     answer: from the llm,
    #     source: doc 의 'source'   <- Document 의 meta data 포함
    #     date: doc 의 'lastmod'  <- Document 의 마지막 수정날짜 정보도 필요.
    # }
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke({
                                "question": question,
                                "context": doc.page_content,
                            }).content,
                "source": doc.metadata['source'],
                "date": doc.metadata['lastmod'],
            }
            for doc in docs
        ]
    }


choose_prompt = ChatPromptTemplate.from_messages([
    ("system", """
        Use ONLY the following pre-existing answers to answer the user's question.

        Use the answers that have the highest score (more helpful) and favor the most recent ones.

        Cite sources and return the sources as it is.

        Answers: {answers}

    """),
    ("human", "{question}"),
])    

def choose_answer(inputs):
    answers = inputs['answers']
    question = inputs['question']
    choose_chain = choose_prompt | llm

    # answers 는 list 다 .  이를 하나의 문자열로 만들어서 체인호출하자.
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
        for answer in answers
    )

    return choose_chain.invoke({
        "question": question,
        "answers": condensed,
    })
    

# ────────────────────────────────────────
# 🍇 file load & cache
# ────────────────────────────────────────
def parse_page(soup):  # soup 는 HTML 가 제거되기 '전' 의 문서.
    header = soup.find("header")  # <header> 요소
    footer = soup.find("footer")  # <footer> 요소

    if header:
        header.decompose()   # 해당 요소를 HTML문서(soup객체) 에서 제거
    
    if footer:
        footer.decompose()

    # <header> 와 <footer> 제거한 나머지 text 리턴
    return (
        str(soup.get_text())
            .replace('\n', " ")            
            .replace('Try Studio', "")
    )


@st.cache_resource(show_spinner="Fetching URL...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )




    loader = SitemapLoader(
        url,
        # filter_urls=[
            # "https://mistral.ai/news/vibe-remote-agents-mistral-medium-3-5",

            # 정규표현식 사용
            # /news/ 포함하는 url 만 가져오기
            # r"^(.*\/news\/).*",

            # /news/ 를 포함하지 않는 url 만 가져오기
            # ?! <- negative lookahead
            # r"^(?!.*\/news\/).*",
        # ],

        # 크롤링한 페이지에 대해 전처리를 할수 있는 함수를 제공해줄수 있다. (BeautifulSoup 사용)
        parsing_function=parse_page,
    )
    loader.max_depth = 3  # 기본값 10
    loader.headers = {'User-Agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Mobile Safari/537.36'}
    docs = loader.load_and_split(text_splitter=splitter)

# Embedding 모델의 입력 초과.
# BadRequestError: Error code: 400 - {'error': {'message': 'Requested 602902 tokens, max
# 300000 tokens per request', 'type': 'max_tokens_per_request', 'param': None, 'code':
# 'max_tokens_per_request'}}

    batch_size = 100  # 적절한 크기로 설정
    substores = []
    for i in range(0, len(docs), batch_size):
        chunk = docs[i: i + batch_size]   # batch_size 만큼씩 embedding 할거다
        vector_store = FAISS.from_documents(documents=chunk, embedding=OpenAIEmbeddings())
        substores.append(vector_store)

    # 여러 store 병합
    vector_store = substores[0]
    for store in substores[1:]:
        vector_store.merge_from(store)  # 병합

    return vector_store.as_retriever()  

# ────────────────────────────────────────
# ⭕ Streamlit 로직
# ────────────────────────────────────────
st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)

st.markdown(
"""
    # SiteGPT
            
    Ask questions about the content of a website.
            
    Start by writing the URL of the website on the sidebar.
"""
)

with st.sidebar:
    url = st.text_input(
        "Write down a URL",
        placeholder="https://example.com",
    )

if url:
    # https://mistral.ai/sitemap.xml
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write down a Sitmap URL")

    else:
        retriever = load_website(url)   
        query = st.text_input("Ask a question to the website") 
        
        # Map Re-Rank Chain 만들기. 두개의 chain 이 필요하다
        # 1.첫번째 chain
        #   모든 개별 Document 에 대한 답변 생성 및 채점 담당
        # 2.두번째 chain
        #   모든 답변을 가진 마지막 시점에 실행된다
        #   점수가 제일 높고 + 가장 최신 정보를 담고 있는 답변들 고른다

        if query:
            chain = {
                "docs": retriever,
                "question": RunnablePassthrough(),
            } | RunnableLambda(get_answers) | RunnableLambda(choose_answer)

            result = chain.invoke(query)
            st.markdown(result.content.replace("$", "\$"))  # 간혹 $가 LaTex 로 렌더링 되서리..