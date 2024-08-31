from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import Depends, FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from .predict import load_model, predict_next_move


model = load_model(Path("static/models/chess_white_splitoutput.pth"))

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    model = load_model(Path("static/models/chess_white_splitoutput.pth"))
    yield
    del model


app = FastAPI(lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def load_index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={})

class BoardFen(BaseModel):
    board_fen: str

@app.post("/board")
async def make_move(board_fen: BoardFen):
    return predict_next_move(board_fen.board_fen, model)

