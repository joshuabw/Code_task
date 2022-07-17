import uvicorn
from fastapi import File, UploadFile, FastAPI
import uvicorn
from fastapi import FastAPI
import numpy as np

import cv2
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from tensorflow.keras.models import model_from_json


app = FastAPI(swagger_ui_parameters={"syntaxHighlight.theme": "obsidian"})

from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

#load model json file  
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

image_size = 160
#function takes image and output prediction
def model_prediction(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (image_size, image_size))
    img = np.dstack([img, img, img])
    img = np.expand_dims(img, axis=0)
    
    #load model from model json file and predict the outcome
    prediction = loaded_model.predict(img)
    prediction_score = np.argmax(prediction, axis=-1)

    return prediction_score



@app.post("/Image Prediction")
async def upload(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix
    #create temp filepath for the image file
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)
    tmp_path = str(tmp_path)


    #fit input image file in prediction model
    prediction = model_prediction(tmp_path)
    prediction = prediction.tolist()
    prediction = jsonable_encoder(prediction)

    file.file.close()
    #tmp_path.unlink()
    #return tmp_path
    return JSONResponse(content=prediction)

    

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)