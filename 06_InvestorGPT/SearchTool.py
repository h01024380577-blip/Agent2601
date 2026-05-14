from dotenv import load_dotenv
load_dotenv()

from langchain_openai.chat_models.base import ChatOpenAI
from langchain_core.tools.simple import Tool
from langchain_core.tools.base import BaseTool
from langchain.agents import create_agent

from pydantic import BaseModel, Field
from typing import Any, Type

from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper

llm = ChatOpenAI(temperature=0.1)

# Pydantic 모델로 tool 의 argument schema 정의
class StockMarketSymbolSearchToolArgsSchema(BaseModel):
    query: str = Field(
        description="The query you will search for. Example query: Stock Market Symbol for Apple Company"
    )

# BaseTool 을 상속받아 커스컨 Tool 정의
class StockMarketSymbolSearchTool(BaseTool):
    name: Type[str] = "StockMarketSymbolSearchTool"   # Tool 이름 💥띄어쓰기 안되요
    description: Type[str] = """
        Use this tool to find the stock market symbol for a company.
        It takes a query as an argument.
        Example query: Stock Market Symbol for Apple Company
        """
    
    args_schema: Type[StockMarketSymbolSearchToolArgsSchema] = StockMarketSymbolSearchToolArgsSchema


    # tool 호출시 실행되는 함수 _run()
    def _run(self, query):
        ddg = DuckDuckGoSearchAPIWrapper()
        return ddg.run(query)  # 검색결과(들) 을 하나의 str 으로 묶어서 리턴