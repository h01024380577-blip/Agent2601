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
def load_df(
    csv_file_path: Annotated[str, Field(description="CSV file path")] = ""
):
    """
    Args:
        csv_file_path (str): csv file to load
    Returns: 
        The DataFrame loaded from the cache.
    """    

    if not csv_file_path:
        _df_cache['df'] = pd.read_csv(r"D:\MCP2601\AgentWork\10_MCP서버개발\121_analytics\data.csv")
    else:
        _df_cache['df'] = pd.read_csv(csv_file_path)

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
    df = _df_cache['df']
    operations = {
        'shape': lambda: df.shape,
        'dtypes': lambda: df.dtypes,
        'missing': lambda: df.isnull().sum(),
        'columns': lambda: list(df.columns),
        'describe': lambda: df.describe(),
    }
    if operation not in operations:
        raise ValueError(f"💥Unsupported operation: {operation}")

    return operations[operation]()


# 특정 컬럼 데이터 확인 (고윳값, 값별 개수) 도구 함수
@mcp.tool(
    name="column_data_check",
    description="Run a column-specific data check operation on the cached DataFrame. Supported operations: unique, value_counts"
)
def column_data_check(
    operation: Annotated[str, Field(description="The kind of column data check to perform (unique, value_counts).")],  # 'unique', 'value_counts'
    column: Annotated[str, Field(description="The name of the column to operate on.")]
):
    """
    Args:
        operation (str): The operation to perform on the column (one of "unique", "value_counts")
        column (str): The name of the column to operate on

    Returns:
        The unique values or value counts of the column
    """    
    df = _df_cache['df']

    if column not in df.columns:
        raise ValueError(f"💥Column '{column}' not found in Dataframe")

    operations = {
        "unique": lambda: list(df[column].unique()),
        "value_counts": lambda: df[column].value_counts()
    }

    if operation not in operations:
        raise ValueError(f"💥Unsupported operation: {operation}")

    return operations[operation]()

# 데이터 전처리 (결측치 제거, 중복제거) 도구
@mcp.tool(
    name="data_preprocess",
    description="Run a basic data preprocessing operation on the cached DataFrame and update the cache. Supported operations: dropna, drop_duplicates"
)
def data_preprocess(
    operation: Annotated[str, Field(description="The preprocessing operation to perform (dropna, drop_duplicates).")]  # "dropna", "drop_duplicates"    
):
    """
    Args:
        operation (str): The preprocessing operation to perform (one of "dropna", "drop_duplicates")

    Returns:
        The DataFrame after preprocessing, updated in the cache.
    """    
    df = _df_cache['df']
    operations = {
        'dropna': lambda: df.dropna(),
        'drop_duplicates': lambda: df.drop_duplicates(),
    }
    if operation not in operations:
        raise ValueError(f"💥Unsupported operation: {operation}")

    result = operations[operation]() 
    _df_cache['df'] = result  # 원본 변경.

    return result


# 컬럼 기반 데이터 필터링 도구 함수
@mcp.tool(
    name="col_data_analysis",
    description="Column-based data analysis. Supported operations: filter_gt (greater than), filter_eq (equal to), filter_lt (less than)"
)
def col_data_analysis(
    operation: Annotated[str, Field(description="The filtering operation to perform (filter_gt, filter_eq, filter_lt).")],  # 'filter_gt', 'filter_eq', 'filter_lt'
    column: Annotated[str, Field(description="The name of the column to filter.")],  # 필터링할 컬럼
    condition_value:  Annotated[float, Field(description="The value to compare against.")]  # 비교할 값
):
    """
    Args:
        operation (str): The filtering operation to perform (one of "filter_gt", "filter_eq", "filter_lt")
        column (str): The name of the column to filter
        condition_value (int): The value to compare against

    Returns:
        The filtered DataFrame
    """
    
    df = _df_cache['df']

    if column not in df.columns:
        raise ValueError(f"💥Column '{column}' not found in Dataframe")

    operations = {
        "filter_gt": lambda: df[df[column] > condition_value],
        "filter_eq": lambda: df[df[column] == condition_value],
        "filter_lt": lambda: df[df[column] < condition_value],
    }

    if operation not in operations:
        raise ValueError(f"💥Unsupported operation: {operation}")

    return operations[operation]()  


# 그룹 기반 데이터 집계 도구 함수
@mcp.tool(
    name="group_data_analysis",
    description="Group-based data analysis. Supported operations: mean, max, sum, count"
)
def group_data_analysis(
    operation: Annotated[str, Field(description="The aggregation operation to perform (mean, max, sum, count).")],     # 'mean', 'max', 'sum', 'count'
    group_column: Annotated[str, Field(description="The name of the column to group by.")],   # 그룹할 컬럼
    target_column: Annotated[str, Field(description="The name of the column to aggregate.")],  # 집계적용할 컬럼
):
    """
    Args:
        operation (str): The aggregation operation to perform (one of "mean", "max", "sum", "count")
        group_column (str): The name of the column to group by
        target_column (str): The name of the column to aggregate

    Returns:
        The result of the group-based aggregation
    """
    
    df = _df_cache['df']

    if group_column not in df.columns:
        raise ValueError(f"💥Group Column '{group_column}' not found in Dataframe")

    if target_column not in df.columns:
        raise ValueError(f"💥Target Column '{target_column}' not found in Dataframe")

    operations = {
        "mean": lambda: df.groupby(group_column)[target_column].mean(),
        "max": lambda: df.groupby(group_column)[target_column].max(),
        "sum": lambda: df.groupby(group_column)[target_column].sum(),
        "count": lambda: df.groupby(group_column)[target_column].count(),
    }

    if operation not in operations:
        raise ValueError(f"💥Unsupported operation: {operation}")

    return operations[operation]()   

# MCP 실행
if __name__ == "__main__":
    mcp.run()
