# Django, Flask, FastAPI

from fastapi import FastAPI
from fastapi.templating import Jinja2Templates

app = FastAPI()


templates = Jinja2Templates(directory="templates")


# 어떤 url 요청 ->  어떠한 응답 : 라우팅 함수로 정의 

@app.get("/")   # HTTP GET 요청 경로 "/" 로 요청이 들어오면 처리하여 응답하는 함수
def read_root():
    # 응답을 리턴
    return {"message": "Hello FastAPI!"}  # 기본적으로 JSON 으로 응답

# 실행
# uvicorn 파일명:앱이름 --reload
# uvicorn 01_Hello:app --reload

from fastapi import Request

@app.get("/user/{username}")
def get_user(username: str, request: Request):
    # return {"username": username}
    return templates.TemplateResponse(request, "username.html", {"username": username})


