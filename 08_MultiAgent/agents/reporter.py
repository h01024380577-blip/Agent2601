# 보고서 작성 에이전트
# 마크다운 형식으로 보고서 작성

from datetime import datetime
from langchain_core.messages import AIMessage

from state import NewsState
from config import Config


class ReportGeneratorAgent:
    """최종 보고서를 생성하는 에이전트"""

    def __init__(self):
        self.name = "Report Generator"

    async def generate_report(self, state: NewsState) -> NewsState:  # 노드로 사용할거다
        """최종 보고서 작성"""
        print(f"\n[{self.name}] 📙보고서 생성 시작...")
        report_parts = []  # 최종보고서에 종합할 각 리포트의 파트들

        current_time = datetime.now().strftime("%Y년 %m월 %d일 %H:%M:%S")

        # 🟪 뉴스헤더
        # ① 모든 카테고리의 뉴스 개수를 합산하여 처리된 총 뉴스 수 계산
        #  total_processed <= 수집 및 처리한 모든 뉴스 개수 저장. 
        #  state.categorized_news는 {"정치": [...], "경제": [...]} 형태인데, 각 카테고리의 
        #  뉴스 리스트 길이를 모두 합하여 총 개수를 구합니다.
        #  제너레이터 표현식 으로 구현 컴프리헨션보다 메모리 효율성이 좋습니다.        
        total_processed = sum(len(v) for v in state.categorized_news.values())
        header = f"""
# Google News 한국 뉴스 AI 요약 리포트

## 기본 정보
- **수집 시간**: {current_time}
- **RSS 소스**: Google News Korea
- **수집 뉴스**: {len(state.raw_news)}건
- **처리 완료**: {total_processed}건
        """
        report_parts.append(header)

        # 🟪통계 섹션
        # ② 각 카테고리별 뉴스 개수 집계
        #  : 카테고리별로 집계 => {"정치": 15, "경제": 12,...}
        category_stats = {
            cat: len(news) for cat, news in state.categorized_news.items()
        }
        total_news = sum(category_stats.values())

        if total_news > 0:
            # 마크다운 테이블 헤더 
            table_header = (
                "| 카테고리 | 뉴스 수 | 비율 |\n|---------|--------|------|\n"
            )            

            # 뉴수 개수가 많은 순으로 정렬하여 테이블의 행 생성
            table_rows = [
                f"| {cat} | {count}건 | {(count / total_news) * 100:.1f}% |"
                for cat, count in sorted(category_stats.items(), key=lambda x: x[1], reverse=True)
                if count > 0
            ]

            stats_table = table_header + "\n".join(table_rows)
            stats_section = f"## 카테고리별 뉴스분류\n\n{stats_table}"
            report_parts.append(stats_section)

        # 🟪카테고리별 뉴스 섹션 생성
        news_sections = []
        for category in Config.NEWS_CATEGORIES:
            news_list = state.categorized_news.get(category)

            if news_list:
                section_header = f"### {category} ({len(news_list)}건)\n"
                # 카테고리별 표시 개수 제한
                display_count = min(len(news_list), Config.NEWS_PER_CATEGORY)

                news_items_str = "\n".join(
                    f"""
#### {i}. {news["title"]}
- **출처**: {news["source"]}
- **발행**: {news.get("published_kst", "")}
- **요약**: {news.get("ai_summary", news["content"])}
- **링크**: [기사 보기]({news["original_url"]})
                    """
                    for i, news in enumerate(news_list[:display_count], 1)
                )

                news_sections.append(f"{section_header}\n{news_items_str}")

        if news_sections:
            report_parts.append(
                "## 카테고리별 주요 뉴스\n\n" + "\n\n---\n\n".join(news_sections)
            )

        # 🟪 에러로그
        if state.error_log:
            errors = "\n".join(f"- {error}" for error in state.error_log)
            report_parts.append(f"## 처리 중 발생한 오류\n\n{errors}")

        # 🟪 푸터 생성
        footer = """## 참고사항
- 이 보고서는 AI(LangGraph + LangChain)를 활용하여 자동으로 생성되었습니다.
- 뉴스 요약은 OpenAI GPT 모델을 사용하여 작성되었습니다.
- 카테고리 분류는 AI가 제목과 내용을 분석하여 자동으로 수행했습니다.
- 상세한 내용은 각 뉴스의 원문 링크를 참조하시기 바랍니다."""
        report_parts.append(footer)

        # 🟪 최종 보고서 조합
        state.final_report = "\n\n---\n\n".join(report_parts)
        state.messages.append(AIMessage(content="최종 보고서가 생성되었습니다"))

        print(f"[{self.name}] 📙보고서 생성 완료...")
        return state