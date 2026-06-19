from typing import Annotated

from fastmcp import FastMCP
from pydantic import Field

mcp  = FastMCP(name="작성규칙")

@mcp.tool(
    name="함수명"
    description="함수의 목적 및 역할을 간단히 설명"
)
def 함수명(
    파라미터1: Annotated[타입, Field(description="파라미터1의 역할 및 설명")]
    # ....
):
    """
    Args:

    파라미터1 (타입): 파라미터1의 역할 및 설명

    Returns:

    반환타입: 반환값의 역할 및 설명    
    """

    # 함수 본문

    return "결과"