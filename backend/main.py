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
    #Image.open(io.BytesIO(file_bytes)).show()
    image = Image.open(data.file)
    num_array = np.array(image)
    #for x in num_array:
    #    for y in  x:
    #        for z in y:
    #            print(z, end=' ')
    #        print()
    
    #image_test = Image.fromarray(num_array)
    #image_test.show()
    return {"result": predict(num_array)}