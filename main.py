from fastapi import FastAPI, Depends, UploadFile, File
from persistence.db import MetaDataBase

app = FastAPI()


@app.get("/ping")
async def ping():
    return {"ping": "pong"}


@app.get("/suggest/tag/{search}")
async def get_autocomplete_suggestions(
    search, db: MetaDataBase = Depends(MetaDataBase)
):
    return {"suggestions": db.get_autocomplete_suggestions(search)}


@app.get("/suggest/image/{search}")
async def get_image_suggestions(search, db: MetaDataBase = Depends(MetaDataBase)):
    return {"suggestions": db.get_image_suggestions(search)}


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
