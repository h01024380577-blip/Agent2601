"""
from contextlib import asynccontextmanager
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

이 import들은 MCP 클라이언트를 구현할 때 자주 등장하는 조합이다.
"""

# Python 표준 라이브러리의 데코레이터입니다. 
# 일반 제너레이터 함수를 비동기 컨텍스트 매니저로 바꿔주는 역할.  (context manager -> with 구문 사용 가능한 객체) 

from contextlib import asynccontextmanager
import asyncio

@asynccontextmanager
async def my_resource():
    print('🧡진입:리소스 준비')
    yield "💚리소스"
    print('💙종료:리소스 정리')

async def test():
    async with my_resource() as r:
        print('🟡 with 시작 --------')
        print(r)
        print('🟡 with 종료---------')

asyncio.run(test())












