from mcp import ClientSession
import asyncio

"""
from mcp import ClientSession
MCP 공식 Python SDK의 핵심 클래스로, 서버와의 통신 세션을 담당합니다. 

내부적으로 read/write 스트림을 받아서 아래와 같은 메소드를 제공

    initialize() — 서버와 핸드셰이크
    list_tools(), call_tool() — 도구 조회/호출
    list_resources(), read_resource() — 리소스 조회
    list_prompts(), get_prompt() — 프롬프트 조회


참고로 FastMCP 에서의 Context와는 다른 객체다, Context는 '서버' 쪽에서 쓰는 거고 
ClientSession은 '클라이언트' 쪽에서 서버를 호출할 때 쓰는 거다.
"""

# 3가지 transport 방식
                                #    MCP의 Streamable HTTP 전송 계층을 사용하기 위한 클라이언트 팩토리 함수.  
from mcp.client.streamable_http import streamable_http_client # streamable http
from mcp.client.stdio import stdio_client # stdio
from mcp.client.sse import sse_client # SSE

import httpx
http_client = httpx.AsyncClient(timeout=30.0)

url = "http://localhost:8000/mcp"

async def test(url):
    async with streamable_http_client(
            url=url,
            http_client=http_client,
        ) as (read_stream, write_stream, get_session_id):

            async with ClientSession(read_stream, write_stream) as session:
                  
                await session.initialize()

                # 툴 목록
                tools_result = await session.list_tools()
                print('🥎', type(tools_result))

                for tool_result in tools_result:
                     print('🌐', type(tool_result), tool_result)
                     if tool_result[0] == 'tools':
                          tools = tool_result[1]
                          for tool in tools:
                               print('🔨', tool)

                print('🟩' * 20)

                # 툴 호출
                call_tool_result = await session.call_tool(name = 'scrape_page_text', arguments={'url': 'https://www.yes24.com/Product/Goods/192475565'})
                print('🟠',
                        type(call_tool_result),
                        call_tool_result,   # <- content 속성에 담겨 있다.  List[Content]
                      )           
                print('💛', len(call_tool_result.content), '개 result')
                print('🟪', call_tool_result.content[0].text[:100])     

asyncio.run(test(url))

"""
2026.06 현재 streamablehttp_client() 는  💢deprecated 됨
MCP 공식 Python SDK에 PR #1177이 머지되면서, 
기존의 httpx_client_factory 콜백 패턴을 버리고 httpx.AsyncClient를 직접 받는 
새로운 streamable_http_client 함수가 추가됐습니다. 
기존 streamablehttp_client는 deprecated 되었지만 하위 호환을 위해 남아있습니다

이름에 헷갈리니 주의!
    streamablehttp_client   (붙여쓰기, 기존)        💢deprecated
    streamable_http_client  (언더스코어, 신규)       ✅권장

새 API가 좋은 이유
팩토리 콜러블 대신 설정된 httpx.AsyncClient를 직접 넘길 수 있고, 
이름도 더 직관적이며, 타임아웃·인증·SSL·헤더 같은 클라이언트 설정에 대한 완전한 제어권을 사용자가 갖게 됩니다    
"""











