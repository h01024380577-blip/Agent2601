from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from pathlib import Path
import uvicorn

from fastapi.staticfiles import StaticFiles  # 정적파일 설정

app = FastAPI()


# 정적 파일 마운트
static_path = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=static_path), name="static")

templates_path = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=templates_path)


@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse(request, "index.html")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)

# 실행
# python 명령으로 실행 가능. 