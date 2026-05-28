import os

class Config:
    """프로젝트 설정 관리 클래스"""

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    MODEL_NAME: str = "gpt-4o"
    MAX_TOKENS: int = 150


    # 현재 파일위치 기준 루트 디렉토리
    ROOT_DIR: str = os.path.dirname(os.path.abspath(__file__))

    # RSS 설정
    # RSS_URL: str = "https://news.google.com/rss?hl=ko&gl=KR&ceid=KR:ko"
    # MAX_NEWS_COUNT: int = 60    

    # LLM 호출을 순차적으로 진행하면 굉장히 시간 걸릴거다.
    # 이를 '동시에 진행' 할수 있도록 
    BATCH_SIZE: int = 10

    # ④ 뉴스를 분류할 카테고리 목록을 정의
    # : 뉴스에 적합한 카테고리를 미리 정의해둡니다. 이 카테고리들은 AI가 뉴스를 분류할 때 사용하는 기준이 됩니다. 
    #   "기타" 카테고리를 마지막에 두어 다른 카테고리에 속하지 않는 뉴스들을 처리할 수 있도록 합니다.
    NEWS_CATEGORIES: list[str] = [
        "정치",
        "경제",
        "사회",
        "문화/연예",
        "IT/과학",
        "스포츠",
        "국제",
        "생활/건강",
        "기타",
    ]    

    NEWS_PER_CATEGORY: int = 30  # 카테고리별 표시한 뉴스 개수

    # 출력파일 저장할 디렉토리
    OUTPUT_DIR: str = f"{ROOT_DIR}/outputs"


    # config 유효성 검사 메소드
    @classmethod
    def validate(cls) -> bool:
        """설정 유효성 검사"""
        if not cls.OPENAI_API_KEY:
            print("💢OpenAI API 키가 설정되지 않았습니다.")
            print("  환경변수 OPENAI_API_KEY를 설정하거나 실행 시 입력하세요.")
            return False
        
        return True
