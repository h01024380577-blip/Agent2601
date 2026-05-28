from typing import Annotated, Any
from pydantic import BaseModel, ConfigDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class NewsState(BaseModel):
    """뉴스 처리 상태를 관리하는 BaseModel"""

    # 우리는 BaseMessage 타입을 사용하나, Pydantic 은 이 타입을 모른다.
    # 임의의 타입을 사용하도록 설정.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # 대화의 이력
    # add_message() 를 사용하면 상태 반환시 '덮어쓰기' 가 아니라 '추가' 동작 하게 될거다.
    messages: Annotated[list[BaseMessage], add_messages] = []
    
    #    RSS 피드에서 수집한 원시 뉴스 데이터(들)을 list[dict] 형태로 저장. 
    #    각 뉴스는 dict 형태로 저장됩니다 {제목:..., 링크:..., 설명:..., 등..}. 
    #    초깃값은 빈 리스트입니다.
    raw_news: list[dict[str, Any]] = []

    #   AI가 요약한 뉴스 데이터를 저장합니다. 
    #   원시 뉴스 에 '요약 정보'가 '추가'된 형태입니다. 
    #   요약 과정을 거친 후의 데이터를 보관합니다.
    summarized_news: list[dict[str, Any]] = []

    # 카테고리별로 분류된 뉴스 저장
    categorized_news: dict[str, list[dict[str, Any]]] = {}

    # 최종 생성된 리포트, 마크다운 형식
    final_report: str = ""

    # 워크플로 실행중 발생하는 에러를 기록
    error_log: list[str] = []   # append() 함수로 직접 추가할거다.

