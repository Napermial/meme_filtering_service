from typing import Union

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.get("/suggestions")
async def get_suggestions(completable: str):
    return {"item_id": item_id, "q": q}