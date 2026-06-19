# 현재 날짜와 시각을 반환하는 MCP 서버


# MCP 서버 코드 구성
# 1. FastMCP 인스턴스를 생성한다.
# 2. MCP 도구로 사용할 함수를 정의하고, 데코레이터로 등록한다.
# 3. 해당 MCP 서버를 클로드 데스크톱에 등록해서 실행한다.


from datetime import datetime

from fastmcp import FastMCP
mcp = FastMCP(name = "Datetime-MCP")  # 서버의 이름지정. <- 서버를 구분하거나 로그 확인에 사용

#  @mcp.tool() : 함수를 MCP 에서 사용할 수 있는 '도구' 로 등록
#      => 외부의 AI 시스템이나 다른 MCP 서버에서도 호출 가능.
@mcp.tool()
def get_current_datetime() -> str:
    """현재 날짜와 시각을 반환합니다"""
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

@mcp.tool()
def add(a: int, b:int) -> int: #MCP 도구의 이름(add)과 입력 스키마
    """Adds two integer numbers together.""" # 도구에 대한 설명
    return a + b

@mcp.tool()
def greet_user(name: str, is_morning: bool = False) -> str: # MCP 도구의 이름(greet_user)과 입력 스키마
    """Greet the user with a personalized message.""" # 도구에 대한 설명
    greeting = "Good morning" if is_morning else "Hello"
    return f"{greeting}, {name}!"


if __name__ == "__main__":
    mcp.run()