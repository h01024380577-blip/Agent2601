import json
import re
import asyncio
from typing import Optional

import httpx
import feedparser
import trafilatura
from bs4 import BeautifulSoup

from utils import convert_gmt_to_kst
from state import NewsState

# 구글 뉴스 API 관련 상수 정의 : 구글 뉴스 서비스와 통신하기 위한 기본 URL들을 상수로 정의합니다.
#  _/DotsSplashUi/data/batchexecute 는 구글의 내부 API 엔드포인트로, 공식적으로 문서화되지 않은 API입니다. 
# KOREA_PARAMS= 는 한국어(hl=ko), 한국 지역(gl=KR), 한국 콘텐츠 ID(ceid=KR:ko)를 지정하는 파라미터입니다.
GOOGLE_NEWS_BASE_URL = "https://news.google.com"
GOOGLE_NEWS_API_URL = f"{GOOGLE_NEWS_BASE_URL}/_/DotsSplashUi/data/batchexecute"
KOREA_PARAMS = "&hl=ko&gl=KR&ceid=KR:ko"

class RSSCollectorAgent:
    """RSS 피드를 수집하는 에이전트"""

    def __init__(self):
        self.name = "RSS Collector"
        # 한국 구글 뉴스 RSS피드 URL 구성
        self.rss_url = f"{GOOGLE_NEWS_BASE_URL}/rss?{KOREA_PARAMS[1:]}"
        self.feed = None

    def load_feed(self) -> None:
        """RSS 피드를 로드합니다."""
        # feedparser를 사용하여 RSS XML 파싱:
        # feedparser는 RSS/Atom 피드를 파싱하는 파이썬 라이브러리입니다. 
        # XML 형식의 RSS 피드를 파싱하여 파이썬 객체로 변환해줍니다. 
        # feedparser가 없다면 httpx를 사용하여 XML을 가져오고, 다시 파싱하는 과정을 거쳐야 하는데 이를 단순하게 해줍니다.
        self.feed = feedparser.parse(self.rss_url)    

    # 뉴스피드의 구글주소URL -> 실제 뉴스 URL 
    async def extract_article_url(self, google_news_url: str) -> Optional[str]:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(google_news_url)
                soup = BeautifulSoup(response.text, "html.parser")

                # 구글 뉴스 페이지에서 c-wiz 컴포넌트 데이터 추출
                #  : c-wiz는 구글이 사용하는 웹 컴포넌트입니다. data-p 속성에는 실제 기사 URL을 얻기 위해 필요한 암호화된 데이터가 들어 있다.
                data_element = soup.select_one("c-wiz[data-p]")
                if not data_element:
                    return None

                # API 요청 페이로드 구성
                # • 구글 내부 API 호출을 위한 페이로드 구성 
                # : 구글의 내부 자료 형식을 디코딩하고, API 호출에 필요한 형태로 변환합니다. 
                # "%.@."는 구글이 사용하는 특수한 플레이스홀더 placeholder 입니다. 
                # headers= 값도 파이썬 프로그램이 아니라 웹브라우저에서 호출하는 것처럼 변경해줍니다.
                raw_data = data_element.get("data-p")
                json_data = json.loads(raw_data.replace("%.@.", '["garturlreq",'))
                payload = {
                    "f.req": json.dumps(
                        [
                            [
                                [
                                    "Fbv4je",
                                    json.dumps(json_data[:-6] + json_data[-2:]),
                                    "null",
                                    "generic",
                                ]
                            ]
                        ]
                    )
                }

                headers = {
                    "content-type": "application/x-www-form-urlencoded;charset=UTF-8",
                    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                }

                # Google 내부 API 호출하여 실제 기사 URL 추출 
                # : API 응답은 중첩된 배열 구조로 되어 있으며, 실제 URL은 [0][2] 위치의 JSON을 파싱한 후 인덱스 1에 위치합니다.
                api_response = await client.post(
                    GOOGLE_NEWS_API_URL, headers=headers, data=payload
                )
                cleaned_response = api_response.text.replace(")]}'", "")
                response_data = json.loads(cleaned_response)
                article_url = json.loads(response_data[0][2])[1]
                return article_url

            except Exception:
                return None

    @staticmethod
    def extract_chosun_content(html_content):
        """조선일보 기사 내용을 특별 처리합니다."""
        # ② 조선일보는 Fusion.globalContent에서 기사 추출
        #  일반적인 텍스트로 이루어진 페이지는 trafilatura를 사용하여 텍스트를 추출할 수 있는데, 
        #  조선일보는 리액트 기반의 Fusion 프레임워크를 사용하고 있어서 기사를 추출할 수가 없었기에, 따로 메서드 제공 ㅜㅜ. 
        #  Fusion 프레임워크는 기사 데이터를 Fusion.globalContent라는 자바스크립트 변수에 저장. 
        #  정규 표현식으로 이 변수의 JSON 데이터를 추출합니다.
        pattern = r"Fusion\.globalContent\s*=\s*({.*?});"
        match = re.search(pattern, html_content, re.DOTALL)

        if match:
            try:
                content_data = json.loads(match.group(1))
                texts = []

                # content_elements 배열에서 텍스트 타입 요소만 추출: 
                #  조선일보의 기사는 여러 요소(이미지, 텍스트, 광고 등)로 구성되어 있습니다. 
                #  이 중에서 type이 "text"인 요소만 필터링하여 순수한 기사 본문만 추출합니다. 
                #  정규표현식에 매칭되는 문자열이 없다면 빈 문자열을 반환합니다. 
                if "content_elements" in content_data:
                    for element in content_data["content_elements"]:
                        if element.get("type") == "text" and "content" in element:
                            texts.append(element["content"])
                return "\n\n".join(texts)
            except json.JSONDecodeError:
                pass
        return ""


    async def parse_entry(self, entry) -> dict[str, Optional[str]]:
        """RSS 피드 항목을 파싱합니다."""
        # 구글 뉴스 URL에 한국 파라미터 추가
        # : RSS 피드에서 제공하는 기본 URL을 브라우저에서 접속을 하면 다시 한국 지역 파라미터가 뒤에 붙는 것을 확인할 수 있습니다. 
        # 기본 URL에 한국 지역 파라미터를 추가하여 한국 버전의 페이지를 요청합니다.
        google_news_url = entry.link + KOREA_PARAMS

        # ④ 실제 기사 URL 추출 및 내용 수집
        # : ①에서 만든 URL을 사용하여 구글 뉴스의 리다이렉션 페이지에서 실제 언론사의 기사 URL을 추출.
        original_url = await self.extract_article_url(google_news_url)
        content = ""

        if original_url:
            # trafilatura를 사용하여 웹페이지 다운로드
            # : trafilatura는 웹 스크래핑에 특화된 라이브러리로, User-Agent 설정, 인코딩 처리 등을 자동으로 처리해줍니다. 
            #   trafilatura.fetch_url(URL)을 사용하면 편하게 URL에서 웹페이지를 내려받을 수 있습니다.
            downloaded = trafilatura.fetch_url(original_url)    
                
            if downloaded:
                # 조선일보는 특별한 파싱 로직 적용: 조선일보는 trafilatura를 사용한 추출 방법이 작동하지 않아 
                # 별도의 extract_chosun_content() 메서드를 사용.
                if (
                    "m.health.chosun.com" not in original_url
                    and "chosun.com" in original_url
                ):
                    content = self.extract_chosun_content(downloaded)
                else:
                    # ⑤ trafilatura로 일반 기사 추출 (한국어 최적화)
                    #   : 한국어 텍스트 추출을 최적화하고, 댓글, 이미지, 링크 등 불필요한 요소를 제외하여 순수한 기사 본문만 추출.
                    content = trafilatura.extract(  # -> str
                        downloaded,
                        include_comments=False,
                        include_images=False,
                        include_links=False,
                        target_language="ko",
                    )

        # 파싱된 뉴스 정보를 딕셔너리로 반환
        # : 다음 에이전트들이 처리하기 편하도록 수집된 모든 정보를 구조화된 딕셔너리 형태로 반환.
        return {
            "title": entry.title,
            "published_kst": convert_gmt_to_kst(entry.published),
            "source": entry.source.get("title", "Unknown"),
            "google_news_url": google_news_url,
            "original_url": original_url,
            "content": content or "",
        }

    async def collect_rss(self, state: NewsState) -> NewsState:  # 나중에 workflow 의 node로 사용될거다
        """RSS 피드를 수집하고 상태를 업데이트합니다."""
        print(f"\n[{self.name}] 🟠 RSS 피드 수집 시작 ")

        try:
            if not self.feed:
                self.load_feed()

            tasks = [self.parse_entry(entry) for entry in self.feed.entries]
            raw_news = await asyncio.gather(*tasks, return_exceptions=True)

            # 수집된 뉴스를 상태객체에 저장. 실패한 결과만 빼고!
            state.raw_news = [
                result
                for result in raw_news
                if not isinstance(result, Exception)
            ]

            print(f"[{self.name}] 🟠총 {len(state.raw_news)} 개의 뉴스 기사 수집 완료!")

        except Exception as e:
            print(f"💢RSS 피드 수집 중 오류 발생: {e}")
            state.error_log.append(f"RSSCollectorAgent: {str(e)}")

        return state
