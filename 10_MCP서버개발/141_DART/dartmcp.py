# 🟦라이브러리 및 환경설정
import OpenDartReader
import pandas as pd
import os
from fastmcp import FastMCP
from typing import Annotated, Literal
from pydantic import Field
from dotenv import load_dotenv
load_dotenv()

mcp = FastMCP("Dart-MCP")
dart = OpenDartReader(os.environ.get("DART_API_KEY"))

# 보고서, 이벤트 코드
REPORT_CODES = [
    '조건부자본증권미상환', '미등기임원보수', '회사채미상환', '단기사채미상환', '기업어음미상환',
    '채무증권발행', '사모자금사용', '공모자금사용', '임원전체보수승인', '임원전체보수유형',
    '주식총수', '회계감사', '감사용역', '회계감사용역계약', '사외이사', '신종자본증권미상환',
    '증자', '배당', '자기주식', '최대주주', '최대주주변동', '소액주주', '임원', '직원',
    '임원개인보수', '임원전체보수', '개인별보수', '타법인출자'
]

EVENT_CODES = [
    '부도발생', '영업정지', '회생절차', '해산사유', '유상증자', '무상증자', '유무상증자', '감자',
    '관리절차개시', '소송', '해외상장결정', '해외상장폐지결정', '해외상장', '해외상장폐지',
    '전환사채발행', '신주인수권부사채발행', '교환사채발행', '관리절차중단', '조건부자본증권발행',
    '자산양수도', '타법인증권양도', '유형자산양도', '유형자산양수', '타법인증권양수', '영업양도',
    '영업양수', '자기주식취득신탁계약해지', '자기주식취득신탁계약체결', '자기주식처분', '자기주식취득',
    '주식교환', '회사분할합병', '회사분할', '회사합병', '사채권양수', '사채권양도결정'
]

# 기업명으로 고유번호 얻기
@mcp.tool(
    name="get_corp_code",
    description="Fetch the corporate code of a company."
)
def get_corp_code(
    corp_name: Annotated[str, Field(description="Corporate name of the company.")]
    ):
    """
    MCP tool for fetching a company's corporate code.
    Args:
        corp_name (str): Corporate name of the company.
    Returns:
        str: Corporate code of the company.
    """    
    return dart.find_corp_code(corp_name)

# 공시정보 - 기업 개황
@mcp.tool(
    name="get_company_overview",
    description="Fetch the general overview information of a company."
)
def get_company_overview(
    corp_code: Annotated[str, Field(description="Corporate code of the company.")]
):
    """
    MCP tool for fetching a company's overview information.
    Args:
        corp_code (str): Corporate code of the company.
    Returns:
        dict: Company overview information.
    """    
    return dart.company(corp_code)

# 🟡 TO Next