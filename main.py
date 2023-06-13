from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
 
from typing import List
 
IMAGEDIR = "images/"
 
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/images", StaticFiles(directory="images"), name="images")
 
@app.get('/', response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
 
@app.post("/upload-files")
async def create_upload_files(request: Request, files: List[UploadFile] = File(...)):
    for file in files:
        contents = await file.read()
        #save the file
        with open(f"{IMAGEDIR}{file.filename}", "wb") as f:
            f.write(contents)
 
    show = [file.filename for file in files]
 
    #return {"Result": "OK", "filenames": [file.filename for file in files]}
    return templates.TemplateResponse("index.html", {"request": request, "show": show})"""
"""  
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import requests

from typing import List

IMAGEDIR = "images/"
MODEL_ENDPOINT = "https://saisuvarna-samplesite-rdhqet06je2.ws-us99.gitpod.io/:8000/"  # Replace with your model's endpoint

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/images", StaticFiles(directory="images"), name="images")


@app.get('/', response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload-files")
async def create_upload_files(request: Request, files: List[UploadFile] = File(...)):
    for file in files:
        contents = await file.read()
        # Save the file
        with open(f"{IMAGEDIR}{file.filename}", "wb") as f:
            f.write(contents)

        # Send the image to model.py for prediction
        with open(f"{IMAGEDIR}{file.filename}", "rb") as image_file:
            files = {"image": image_file}
            response = requests.post(MODEL_ENDPOINT, files=files)
            prediction = response.json()
            print(prediction)  # Replace with your desired processing of the prediction

    show = [file.filename for file in files]

    return templates.TemplateResponse("index.html", {"request": request, "show": show})
"""
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import tensorflow as tf

# Initialize FastAPI app and template directory
app = FastAPI()
templates = Jinja2Templates(directory="templates")
#app.mount("/static", StaticFiles(directory="static"), name="static")

# Load your pre-trained model
model = tf.keras.models.load_model('C:/Users/HI/Downloads/sequential')

# Define an endpoint to handle image uploads and perform predictions
@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    # Read and preprocess the uploaded image
    contents = await file.read()
    # Perform any necessary preprocessing steps on the image
    # ...

    # Make predictions using the loaded model
    # ...

    # Return the predicted results to the template
    return templates.TemplateResponse("result.html", {"request": request, "predictions": predictions})

# Define other endpoints and routes as needed
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Run the FastAPI application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)"""

