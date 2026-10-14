import io, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import app as medical_app
from PIL import Image

def test_home_page_loads():
    r=medical_app.app.test_client().get("/")
    assert r.status_code==200 and b"Medical Image Analysis" in r.data

def test_post_without_file():
    r=medical_app.app.test_client().post("/",data={})
    assert r.status_code==200 and b"Choose an image first" in r.data

def test_invalid_image_rejected():
    r=medical_app.app.test_client().post("/",data={"image":(io.BytesIO(b"bad image"),"fake.jpg")},content_type="multipart/form-data")
    assert b"valid image file" in r.data

def test_preprocessing_shape_and_range():
    img=Image.new("L",(80,100),color=128); b=io.BytesIO(); img.save(b,format="PNG")
    arr=medical_app.preprocess_image(b.getvalue())
    assert arr.shape==(1,256,256,1) and 0<=arr.min()<=arr.max()<=1
