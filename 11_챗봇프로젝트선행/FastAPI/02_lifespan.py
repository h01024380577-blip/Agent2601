from fastapi import FastAPI, Request


def lifespan(app: FastAPI):
    """FastAPI 애플리케이션 생명주기 동안 관리할 객체 코드 작성"""

    print('🔵 FastAPI 앱 시작')
    # MCP Client - MCP Sever 연결
    # MCP Server 의 tool 들을 사용한 Langchain Agent 생성
    
    yield
    print('🟡 FastAPI 앱 종료')

app = FastAPI(lifespan=lifespan)