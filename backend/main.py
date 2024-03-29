from fastapi import FastAPI, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
from model import predict

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials = True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post('/')
async def root(data: UploadFile = Form(...)):
    print(data.filename, data.content_type)
    image = Image.open(data.file).transpose(Image.FLIP_LEFT_RIGHT)
    num_array = np.array(image)
    return {"result": predict(num_array)}
