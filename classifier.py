import os

from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient
from roboflow import Roboflow
from ultralytics import YOLO


def inference_using_api(img_path):
    CLIENT = InferenceHTTPClient(api_url='https://detect.roboflow.com', api_key=API_KEY)

    result = CLIENT.infer(img_path, model_id='washinglablerecognition-0yaja/2')
    return result


def download_dataset():
    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace('test-ah8ju').project('washinglablerecognition-0yaja')
    version = project.version(4)
    version.download('yolov8')


def train_yolo():
    model = YOLO('yolov8n.pt')
    data_path = './washingLableRecognition-4/data.yaml'
    model.train(data=data_path, epochs=500, imgsz=640)


def test_trained_model(img_path):
    model = YOLO('./model/best.pt')
    results = model(img_path)
    return results


load_dotenv(override=True)
API_KEY = os.getenv('API_KEY')

sample_img = 'data/kin-phinf.pstatic.net_20230626_19_1687746560299AzJ9L_JPEG_1687746560272.jpeg'
results = test_trained_model(sample_img)
