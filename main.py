from fastapi import FastAPI, Depends, UploadFile, File
from persistence.db import MetaDataBase
from fastapi.middleware.cors import CORSMiddleware
import base64

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    db = MetaDataBase()
    yield db


@app.get("/ping")
async def ping():
    return {"ping": "pong"}


@app.get("/suggest/tag/{search}")
async def get_autocomplete_suggestions(
    search, db: MetaDataBase = Depends(get_db)
):
    return {"suggestions": db.get_autocomplete_suggestions(search)}


@app.get("/suggest/image/{search}")
async def get_image_suggestions(search, db: MetaDataBase = Depends(get_db)):
    encoded_images = []
    for i, image_path in enumerate(db.get_image_suggestions(search)):
        if i < 12:
            with open(image_path, "rb") as f:
                encoded_string = base64.b64encode(f.read())
                encoded_images.append(encoded_string)
    return {"suggestions": [{"path": image} for image in encoded_images]}


@app.post("/image/upload")
async def upload_image_to_database(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        with open(file.filename, "wb") as f:
            f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()

    return {"message": f"Successfuly uploaded {file.filename}"}
