import asyncio
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from state import NewsState
from config import Config

class NewsSummarizerAgent:
    """뉴스를 요약하는 에이전트"""

    def __init__(self, llm: ChatOpenAI):
        self.name = "News Summarizer"
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """
                    당신은 전문 뉴스 요약 전문가입니다. 
                    주어진 뉴스를 핵심만 간결하게 2-3문장으로 요약해주세요.
                    - 사실만을 전달하고 추측은 피하세요
                    - 중요한 숫자나 날짜는 포함하세요
                    - 명확하고 이해하기 쉽게 작성하세요
            """),
            ("human", "제목: {title}\n내용: {content}\n\n위 뉴스를 2-3문장으로 요약해주세요:"),
        ])

    async def summarize_single_news(self, news_item: Dict[str, Any]) -> Dict[str, Any]:
        """단일 뉴스 요약 (오류 발생 시 원본 내용 반환)"""
        content = news_item.get("content", "")

        try:
            # 최소 컨텐츠 길이 검증
            if not content or len(content) < 50:
                return {**news_item, "ai_summary": content}

            # 체인
            chain = self.prompt | self.llm
            summary_response = await chain.ainvoke({
                "title": news_item["title"],
                "content": content[:500],
            })

            summary = summary_response.content.strip()

            # LLM 이 간혹 '비어있는 문자열' 리턴하는 경우도 있다. 그래서  summary or content
            return {**news_item, "ai_summary": summary or content}
        
        except Exception as e:
            err_message = f"  [{self.name}] 💥요약 오류 (Title: {news_item['title']}): {str(e)[:50]}..."
            print(err_message)            

            # 오류 발생시 원본 사용
            return {**news_item, "ai_summary": content}
        
    
    async def summarize_news(self, state: NewsState) -> NewsState:   
        """모든 뉴스를 비동기로 요약"""
        # workflow 의 node 로 사용할거다 

        print(f"\n[{self.name}] 🌐뉴스 요약 시작...")

        batch_size = Config.BATCH_SIZE
        summarized_news = []
        raw_news = state.raw_news
        total_news = len(raw_news)        

        # 배치 단위 순차 처리로 API 호출 부하 분산.\
        #   : 수십 개의 뉴스를 한 번에 처리하면 API에 과도한 부하를 주거나 속도 제한에 걸릴 수 있습니다. 
        #   특히 LLM의 경우 많은 요청을 한 번에 보내면 429 too many requests 에러가 나면서 
        #   처리가 안 되는 경우가 종종 있습니다. 이 코드는 전체 뉴스 목록을 batch_size만큼의 작은 묶음(배치)으로 나눕니다. 
        #   각 배치 내의 뉴스들은 asyncio.gather()를 통해 비동기적으로 동시에 처리되어 효율성을 높이고, 
        #   배치 단위로 '순차 처리'하여 API 서버의 부하를 조절합니다.

        for i in range(0, total_news, batch_size):
            batch = raw_news[i: i + batch_size]

            batch_num = i // batch_size + 1
            total_batches = (total_news + batch_size - 1) // batch_size
            print(f"  배치 {batch_num}/{total_batches} 처리 중...")

            tasks = [self.summarize_single_news(news) for news in batch]
            batch_results = await asyncio.gather(*tasks)
            summarized_news.extend(batch_results)


        # 상태 업데이트
        state.summarized_news = summarized_news
        state.messages.append(
            AIMessage(content=f"{len(summarized_news)}개의 뉴스 요약을 완료했습니다.")
        )

        print(f"[{self.name}] 🌐요약 완료\n")
        return state