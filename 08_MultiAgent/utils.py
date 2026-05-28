# 수집된 원시 뉴스 데이터에서 필요한 텍스트만 깔끔하게 뽑아낼 전처리 함수들

from datetime import datetime, timedelta
import re

def clean_html(html_text: str) -> str:
    """HTML 태그 제거"""

    if not html_text: 
        return ""
    
    # 정규표현식 사용 HTML 태그 제거:  <태그명>내용</태그명> 
    clean_text = re.sub("<.*?>", "", html_text)

    # 연속된 공백은 하나의 공백으로 정리
    clean_text = re.sub(r"\s+", " ", clean_text).strip()

    return clean_text


def truncate_text(text: str, max_length: int = 500) -> str:
    """텍스트를 적절한 길이로 자르기"""
    # 긴~ 뉴스 내용을 미리보기로 보여줄때 사용.
    if not text or len(text) <= max_length:
        return text
    
    return text[:max_length] + "..."

# RSS 피드의 GMT 시간을 KST 으로 변환
# RSS 피드의 날짜는 보통 RFC-822 형식(예시: "Mon, 25 Dec 2023 10:30:00 GMT")으로 제공. ← 끝에 시간대 정보가 포함된다.
# 여기에 9시간 더해주면 한국시간 된다.
def convert_gmt_to_kst(gmt_time_str: str) -> str:
    """GMT 시간을 KST로 변환합니다."""
    KST_OFFSET_HOURS = 9
    gmt_time = datetime.strptime(gmt_time_str, "%a, %d %b %Y %H:%M:%S GMT")
    kst_time = gmt_time + timedelta(hours=KST_OFFSET_HOURS)
    return kst_time.strftime("%Y-%m-%d %H:%M:%S")  # "2025-07-08 10:02:00" 형태로 리턴




# 테스트
if __name__ == "__main__":
    # print(clean_html("        <div>    호랑이가 <b>담을</b>       넘어간다</div>"))
    print(convert_gmt_to_kst("Mon, 25 Dec 2023 10:30:00 GMT"))