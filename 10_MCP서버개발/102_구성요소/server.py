from fastmcp import FastMCP
from pydantic import Field

mcp = FastMCP()

# 아래 url 은 resource 의 고유식별자 역할.
# 함수의 이름이 리소스 이름
# docstring 이 리소스 설명
@mcp.resource("resource://greeting")
def get_greeting() -> str:
    """간단한 이사말 메세지를 제공합니다."""
    return "안녕하세요! FastMCP 리소스입니다"

# 지연 로딩(Lazy Loading): 데코레이터를 지정한 함수(예: get_greeting)는 클라이언트가 해당 리소스 URI를 명시적으로 요청할 때에만 실행됩니다. 이는 불필요한 리소스 로딩을 방지하고 효율성을 높입니다.


# 리소스의 메타데이터
@mcp.resource(
    uri="data://app-settings",  #고유 리소스 주소
    name="AppSettings", #사용자 친화적 이름
    description="애플리케이션 설정 정보를 JSON으로 제공합니다.", #LLM용 설명
    mime_type="application/json", #데이터 형식: JSON
    tags={"settings", "config","public"} #분류 태그
)
def load_app_settings() -> dict:
    """내부 doctring 보다 리소스의 description 이 우선된다"""
    return {
        "theme": "light",
        "version":"1.2.1",
        "options": ["tools", "resources", "notifications"],
    }


@mcp.prompt(
    name="analyze_data_request", #프롬프트 명칭
    description="Generates a data analysis request with user-specified options.", #설명
    tags={"analysis", "data"} #태그        
)
def data_analysis_prompt(
    data_uri: str = Field(description="Resource URI containing the target dataset."),
    analysis_type: str = Field(default="summary", description="Desired analysis type.")
) -> str:
    """이 독스트링은 description이 있을 때 무시됨"""
    return f"Analyze the dataset at {data_uri} using the '{analysis_type}' method."
    