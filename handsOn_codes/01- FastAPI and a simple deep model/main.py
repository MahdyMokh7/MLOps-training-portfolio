from fastapi import FastAPI, File
from fastapi.responses import StreamingResponse
from model import detect_model, add_bboxs_on_img, get_bytes_from_image, get_image_from_bytes

# create a FastAPI "instance"
app = FastAPI(title="Object Detection FastAPI")


# Define path operation decorator
@app.get("/")
def root():  # define the path operation function
    return {"message": "Hello to the first Hands on"}


# Define path operation decorator
@app.post("/img_object_detection")
def img_object_detection(file: bytes = File(...)):  # define the path operation function
    """
    Object Detection from an image, and plot bbox on image

    Args:
        file (bytes): The image file in bytes format.
    Returns:
        Image: Image in bytes with bbox annotations.
    """
    # get image from bytes
    input_image = get_image_from_bytes(file)

    # model predict
    predict = detect_model(input_image)

    # add bbox on image
    final_image = add_bboxs_on_img(image=input_image, predict=predict)

    # Convert image to byte format
    content = get_bytes_from_image(final_image)

    # return bytes format image as streaming response
    return StreamingResponse(content=content, media_type="image/jpeg")
