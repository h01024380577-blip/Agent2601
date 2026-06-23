from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from tavily import TavilyClient
from langchain_core.tools import tool

from datetime import datetime
import json
import os
absolute_path = os.path.abspath(__file__) # 현재 파일의 절대 경로 반환
current_path = os.path.dirname(absolute_path) # 현재 .py 파일이 있는 폴더 경로

# RAG를 위한 설정
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# 오픈AI Embedding 설정
embedding = OpenAIEmbeddings(model='text-embedding-3-large')

# 크로마 DB 저장 경로 설정
persist_directory = f"{current_path}/data/chroma_store"

# Chroma 객체 생성
vectorstore = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)


# @tool 데코레이터는 함수를 랭체인이나 랭그래프에서 도구로 등록할 때 사용합니다.
# 이렇게 하면 해당 함수는.invoke()를 통해 호출할 수 있게 됩니다.

# 이때! doctring 으로 함수에 대한 설명을 작성해야 오류가 발생하지 않습니다.
@tool
def web_search(query: str):
    """
    주어진 query에 대해 웹검색을 하고, 결과를 반환한다.

    Args:
        query (str): 검색어

    Returns:
        dict: 검색 결과
    """
    client = TavilyClient()

    content = client.search(
        query, 
        search_depth='advanced',  # "basic"(기본), "advanced" 는 더 관련성 높은 결과를 찾기 위해 크레딧을 요청당 2개 사용.
        include_raw_content=True,  # 검색된 페이지의 '전문' 가져오기
    )

    results = content['results']

    # tavily 가 간혹 raw_content (전문) 을 못 읽어오는 경우 처리
    for result in results:
        if result['raw_content'] is None:
            try:
                result['raw_content'] = load_web_page(result['url'])
            except Exception as e:
                print(f"💥Error loading page: {result['url']}")
                print(e)
                # load_web_page가 실패하면 content의 원래 값을 그대로 raw_content에 넣도록 처리
                result["raw_content"] = result["content"]

    resources_json_path = f'{current_path}/data/resources_{datetime.now().strftime('%Y_%m%d_%H%M%S')}.json'
    with open(resources_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    return results, resources_json_path   # '검색결과' 와 'JSON 파일 경로' 리턴


def web_page_to_document(web_page):
    # tavily 검색결과의 raw_content 와 content 중 정보가 많은 것을 
    # Document 의 page_content 로 한다
    if len(web_page['raw_content']) > len(web_page['content']):
        page_content = web_page['raw_content']
    else:
        page_content = web_page['content']    

    # Document 로 변환
    document = Document(
        page_content=page_content,
        metadata={
            'title': web_page['title'],
            'source': web_page['url'],
        }
    )

    return document


# 검색결과 저장된 json 파일을 읽어 각 웹페이지 정보를 Document 로 변환
def web_page_json_to_documents(json_file):
    with open(json_file, "r", encoding='utf-8') as f:
        resources = json.load(f)

    documents = []

    for web_page in resources:
        document = web_page_to_document(web_page)
        documents.append(document)

    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=100):
    print('Splitting documents...')
    print(f"{len(documents)}개의 문서를 {chunk_size}자 크기로 중첩 {chunk_overlap}자로 분할합니다.\n")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    splits = text_splitter.split_documents(documents)

    print(f"총 {len(splits)}개의 문서로 분할되었습니다.")
    return splits    


def documents_to_chroma(documents, chunk_size=1000, chunk_overlap=100):
    print("Documents를 Chroma DB에 저장합니다.")

    # documents의 url 가져오기
    urls = [document.metadata['source'] for document in documents]

    # 이미 vectorstore 에 저장된 urls 가져오기
    stored_metadatas = vectorstore._collection.get()['metadatas']
    stored_web_urls = [metadata['source'] for metadata in stored_metadatas]

    # 기존 벡터DB 에 없어던 새로운 urls 만 남기기
    new_urls = set(urls) - set(stored_web_urls)

    # 새로운 urls 에 대한 Document 들만 남기기
    new_documents = []

    for document in documents:
        if document.metadata['source'] in new_urls:
            new_documents.append(document)
            print(document.metadata)

    # new_documents 를 Chroma DB 에 저장
    splits = split_documents(new_documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # ChromaDB 에 저장
    if splits:
        vectorstore.add_documents(splits)
    else:
        print("새로운 urls 들이 없었습니다")


def add_web_pages_json_to_chroma(json_file, chunk_size=1000, chunk_overlap=100):
    documents = web_page_json_to_documents(json_file)
    documents_to_chroma(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )



def load_web_page(url: str):
    loader = WebBaseLoader(url, verify_ssl=False)
    content = loader.load()  # List[Document]

    raw_content = content[0].page_content.strip()

    # 과도한 탭이나 줄바꿈 정리
    while '\n\n\n' in raw_content or '\t\t\t' in raw_content:
        raw_content = raw_content.replace('\n\n\n', '\n\n').replace('\t\t\t', '\t\t')

    return raw_content

@tool
def retrieve(query: str, top_k: int=5):
    """
    주어진 query에 대해 벡터 검색을 수행하고, 결과를 반환한다.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    retrieved_docs = retriever.invoke(query)

    return retrieved_docs

if __name__ == "__main__":
    # results, resources_json_path = web_search.invoke("2026년 한국 영화 시장 전망")
    # print(resources_json_path, '에 저장')
    # print(results[0])

# tavily 검색결과
# {
#     'results' : [
#         {
#             'url': ...
#             'content': ...
#             'score': ...
#             'raw_content': ...  <- tavily 가 종종 이 내용을 읽어오지 못하는 경우도 있다.
#         }
#     ]
# }

    # result = load_web_page('https://m.cafe.daum.net/dobongbak/O1j9/778?svc=TOPRANK')
    # print(result)

    # documents = web_page_json_to_documents(f"{current_path}/data/resources_2026_0605_194706.json")
    # splits = split_documents(documents)
    # print(splits[-3:])

    add_web_pages_json_to_chroma(f"{current_path}/data/resources_2026_0605_213436.json")

    # retrieved_docs = retrieve.invoke({"query": "한국 경제 위험 요소"})
    # print(retrieved_docs)


    # results, resources_json_path = web_search.invoke("HYBE 와 JYP 비교")

    retrieved_docs = retrieve.invoke({"query": "HYBE 의 기업문화"})
    print(retrieved_docs)
