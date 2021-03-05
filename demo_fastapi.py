from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import Optional
import cv2
import numpy as np
from PIL import Image, ExifTags
from tensorflow.python.keras.models import load_model

result = ["Bệnh đạo ôn", "Bệnh đốm nâu", "Bình thường", "Bệnh khô vằn", "Bệnh Tungro"]
model = load_model('best_model.h5')
app = FastAPI()



@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
async def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}

@app.post("/file")
async def postfile(file: UploadFile = File(...)):
    img = Image.open(file.file)
    img = np.array(img)
    image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (128,128))
    image = image/255.0
    image = np.reshape(image, (1,128,128,3))
    pre = model.predict(image)
    return {
        "loaibenh": str(result[np.argmax(pre)]),
        "cachchuabenh": ""
    }
def autoRotateImage(img):
    for orientation in ExifTags.TAGS.keys():
        if ExifTags.TAGS[orientation]=='Orientation':
            break
    
    exif = img._getexif()
    if (exif!=None):
        try:
            if exif[orientation] == 3:
                img=img.rotate(180, expand=True)
            elif exif[orientation] == 6:
                img=img.rotate(270, expand=True)
            elif exif[orientation] == 8:
                img=img.rotate(90, expand=True)
        except:
            return img
    return img