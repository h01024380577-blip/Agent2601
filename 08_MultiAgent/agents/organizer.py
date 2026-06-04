# 뉴스를 카테고리별로 분류하는 에이전트

import asyncio
from typing import Dict, Any, Tuple
from collections import defaultdict
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from state import NewsState
from config import Config

class NewsOrganizerAgent:
    """뉴스를 카테고리별로 정리하는 에이전트"""

    def __init__(self, llm: ChatOpenAI):
        self.name = "News Organizer"
        self.llm = llm

        self.categories = Config.NEWS_CATEGORIES

        system_prompt = f"""
            당신은 뉴스 분류 전문가입니다.
            주어진 뉴스를 다음 카테고리 중 하나로 정확히 분류해주세요:
            {", ".join(Config.NEWS_CATEGORIES)}
            
            반드시 위 카테고리 중 하나만 선택하고, 카테고리 값만 반환하세요.
            """

        self.categorize_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "제목: {title}\n요약: {summary}\n\n이 뉴스의 카테고리:"),
        ])

        self.chain = self.categorize_prompt | self.llm



    async def categorize_single_news(self, news_item: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:        
        """단일 뉴스의 카테고리 분류"""

        # LLM 비동기 호출로 뉴스 분류
        response = await self.chain.ainvoke({
            "title": news_item['title'],
            "summary": news_item.get('ai_summary', news_item['content']),
        })

        category = response.content.strip()
        return category, news_item
    
    async def organize_news(self, state: NewsState) -> NewsState:   # 노드로 사용할거다
        """모든 수집한 뉴스(들)을 카테고리별로 정리"""
        print(f"\n[{self.name}] 🎃뉴스 분류 시작...")

        summarized_news = state.summarized_news
        total_news = len(summarized_news)
        batch_size = Config.BATCH_SIZE

        # 분류된 뉴스 저장을 위한 dict:
        # defaultdict(list)  : 키가 없을때 자동으로 '빈 리스트' 생성하여 KeyError 방지
        categorized = defaultdict(list)

        # 배치 처리 루프
        for i in range(0, total_news, batch_size):
            batch = summarized_news[i: i + batch_size]

            batch_num = i // batch_size + 1
            total_batches = (total_news + batch_size - 1) // batch_size  # 전체 배치개수
            print(f"   배치 {batch_num} / {total_batches} 분류 중...")

            # 비동기 분류 작업 생성
            tasks = [self.categorize_single_news(news) for news in batch]

            # 모든 분류 작업 병렬(동시) 실행
            # ↓ results 는 List[Tuple]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    print(f"     분류 작업 실패: {result}")
                    continue

                category, news_item = result

                # 반환된 카테고리 유효성 검사
                # LLM 이 (간혹) 원하지 않는 값을 출력하는 경우도 있을수 있다. -> "기타" 로 분류
                if category in Config.NEWS_CATEGORIES:
                    categorized[category].append(news_item)
                else:
                    categorized["기타"].append(news_item)

        print("\n  카테고리별 분포:")
        for category in self.categories:
            count = len(categorized.get(category, []))
            if count > 0:
                print(f"     {category}: {count}건")

        # 분류결과를 state 객체에 저장
        state.categorized_news = dict(categorized)
        state.messages.append(
            AIMessage(content=f"뉴스를 {len(categorized)}개 카테고리로 분류했습니다.")
        )

        print(f"[{self.name}] 🎃분류 완료\n")
        return state




























