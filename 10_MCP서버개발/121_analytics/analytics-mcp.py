from fastmcp import FastMCP
import pandas as pd
from typing import Annotated
from pydantic import Field

mcp = FastMCP("Analytics-MCP")

# MCP 서버에서 파이르이 경로는 '절대경로' 사용 
_df_cache = {}
_df_cache['df'] = pd.read_csv(r"D:\MCP2601\AgentWork\10_MCP서버개발\121_analytics\data.csv")

# ---------------------------------------------------------------
# 3. 데이터프레임 불러오기 도구 정의
# 앞에서 정의한 _df_cache에 저장된 데이터프레임을 불러옵니다. 만약 캐시에 데이터프레임이 없다면 에러를 발생시킵니다.
# ---------------------------------------------------------------
@mcp.tool(name="load_df", description="Load the Dataframe from the cache.")
def load_df():
    """
    Args:
        None
    Returns: 
        The DataFrame loaded from the cache.
    """    

    if "df" not in _df_cache:
        raise ValueError("No DataFrame found in cache. Please save a DataFrame with save_df first.")
    
    return _df_cache['df']

# ---------------------------------------------------------------------
# 기본 데이터 확인

# 다음은 캐시에 저장된 데이터프레임에 대해 다양한 기본 정보
# (형태, 데이터 타입, 결측치 개수, 칼럼명, 요약 통계 등)를 조회할 수 있는
# MCP 도구를 정의한 것입니다.
# 사용자가 원하는 분석 종류를 operation 파라미터로 지정하면
# 작업을 수행한 후 결과를 반환합니다.
# 지원하지 않는 operation을 입력할 경우 에러가 발생합니다.
# ---------------------------------------------------------------------

def basic_data_check(
    operation: Annotated[str, Field(description="The kind of basic data check to perform (shape, dtypes, missing, columns, describe).")]
):
    """
    Args:
        operation (str): The kind of basic data check to perform (one of "shape", "dtypes", "missing", "columns", "describe")

    Returns:
        The result of the requested data check operation.
    """
    # 🔷TODO 

