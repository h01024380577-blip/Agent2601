# FastAPI 서버
from pathlib import Path
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

# 랭체인, 랭그래프
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# MCP 클라이언트 
from contextlib import asynccontextmanager
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession
import httpx
from mcp.client.streamable_http import streamable_http_client

# 환경변수
from dotenv import load_dotenv
load_dotenv()



app = FastAPI()

# chat_agent.py 파일의 위치를 기준으로 정적 파일 마운트
static_path = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=static_path), name="static")

# chat_agent.py 파일의 위치를 기준으로 templates 디렉토리의 절대 경로를 계산하여 설정
templates_path = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=templates_path)

# 메인 페이지 라우팅
@app.get("/")
async def read_root(request: Request):
    """메인 채팅 페이지 렌더링"""
    return templates.TemplateResponse(request, "index.html")


if __name__ == "__main__":
    uvicorn.run("chat_agent:app", 
                host="0.0.0.0",  # 127.0.0.1    localhost
                port=8001,    # 현재  MCP server 가 port 8000 을 사용하기에 다른 port 사용
                reload=True)