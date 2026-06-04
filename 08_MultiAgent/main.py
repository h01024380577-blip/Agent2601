from dotenv import load_dotenv
load_dotenv()

import os
import logging
import asyncio
from datetime import datetime
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from workflow import create_news_workflow
from config import Config
from state import NewsState


async def main():
    """Google News AI 멀티에이전트 시스템의 메인 실행 함수"""
    print(
        """
Google News AI 멀티에이전트 시스템
RSS 수집 → AI 요약 → 카테고리 분류 → 리포트 생성
"""
    )    

    try:
        # 설정 유효성 검사
        if not Config.validate():
            raise ValueError("API키가 설정되지 않았습니다. .env파일을 확인해주세요")

        print("\n" + "=" * 60)
        print("뉴스 처리 시작")
        print("=" * 60)

        llm = ChatOpenAI(
            model=Config.MODEL_NAME,
            max_tokens=Config.MAX_TOKENS,
            api_key=Config.OPENAI_API_KEY
        )

        app = create_news_workflow(llm)

        # 워크플로우 실행
        initial_state = NewsState(
            messages = [HumanMessage(content="Google News RSS 처리를 시작합니다.")]
        )
        final_state = await app.ainvoke(initial_state)

        if not final_state.get("final_report"):
            print("\n생성된 보고서가 없습니다")
            return

        # 최종 보고서 저장
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(Config.OUTPUT_DIR, f"news_report_{timestamp}.md")    

        with open(filename, "w", encoding="utf-8") as f:
            f.write(final_state["final_report"])

        print("\n" + "=" * 60)
        print("처리 완료")
        print("=" * 60)
        print(f"\n보고서가 저장되었습니다: {filename}")
        print(f"처리된 뉴스: {len(final_state.get('summarized_news', []))}건")
        print("\n보고서 미리보기:")
        print("-" * 60)
        print(final_state["final_report"][:500] + "...")


    except KeyboardInterrupt:
        print('\n\n사용자에 의해 중단되었습니다')
    except Exception as e:
        print(f"\n오류발생: {e}")

if __name__ == "__main__":
    asyncio.run(main())