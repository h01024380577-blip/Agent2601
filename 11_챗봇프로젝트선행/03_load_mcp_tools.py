# pip install langchain-mcp-adapters

from langchain_mcp_adapters.tools import load_mcp_tools
"""
랭체인의 load_mcp_tools() 함수
   'MCP 도구'를  → 'LangChain 도구'로 변환

   MCP 서버가 제공하는 도구(tool)들을 
   LangChain/LangGraph 에이전트에서 바로 쓸 수 있는 
   LangChain Tool 객체로 변환해주는 함수.

MCP 서버는 자체 프로토콜로 도구 목록과 스키마를 노출하는데, 
그러나, LangChain 에이전트는 LangChain의 BaseTool(주로 StructuredTool) 형태를 기대합니다. 
LangChain 의 load_mcp_tools()가 그 사이를 이어주는 역할을 합니다 

이를통해 LangChain은 'MCP 도구'를 'LangChain 도구'로 변환해서 
어떤 LangChain 에이전트나 워크플로우에서도 바로 쓸 수 있게 만듭니다

"""
# 기본사용 패턴
# 가장 단순한 형태는 이미 연결된 ClientSession을 넘기는 것입니다.
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from langchain.agents import create_agent
import asyncio

import httpx
http_client = httpx.AsyncClient(timeout=30.0)

async def test(url):
    async with streamable_http_client(
        url=url,
        http_client=http_client,
    ) as (read_stream, write_stream, get_session_id):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # LangChain tool <- MCP tool
            tools = await load_mcp_tools(session) # -> List[BaseTool]

            print('💚', type(tools), len(tools), '개 Tool들')

            for tool in tools:
                print('🔨', type(tool), tool)

            # Langchain Agent 생성
            agent = create_agent("gpt-5-mini", tools)

            result = await agent.ainvoke({"messages": "오늘의 명언을 알려줘"})
            print('🟣', result['messages'][-1].content)


url = "http://localhost:8000/mcp"
asyncio.run(test(url))

