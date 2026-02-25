from io import BytesIO

from PIL import Image
import io
import pandas as pd
import numpy as np

from ultralytics import YOLO
from ultralytics.yolo.utils.plotting import Annotator, colors

# Initialize the model
object_detection_model = YOLO("./models/yolov8n.pt")


def get_image_from_bytes(binary_image: bytes) -> Image:
    """Convert image from bytes to PIL RGB format

    Args:
        binary_image (bytes): The binary representation of the image

    Returns:
        PIL.Image: The image in PIL RGB format
    """
    input_image = Image.open(io.BytesIO(binary_image)).convert("RGB")
    return input_image


def get_bytes_from_image(image: Image) -> BytesIO:
    """
    Convert PIL image to Bytes

    Args:
    image (Image): A PIL image instance

    Returns:
    bytes : BytesIO object that contains the image in JPEG format with quality 85
    """
    return_image = io.BytesIO()
    image.save(return_image, format='JPEG', quality=85)  # save the image in JPEG format with quality 85
    return_image.seek(0)  # set the pointer to the beginning of the file
    return return_image


def transform_predict_to_df(results: list, labels_dict: dict) -> pd.DataFrame:
    """
    Transform predict from yolov8 (torch.Tensor) to pandas DataFrame.

    Args:
    results (list): A list containing the predict output from yolov8 in the form of a torch.Tensor.
    labels_dict (dict): A dictionary containing the labels names, where the keys are the class ids and the values
    are the label names.

    Returns: predict_bbox (pd.DataFrame): A DataFrame containing the bounding box coordinates, confidence scores and
    class labels.
    """
    # Transform the Tensor to numpy array
    predict_bbox = pd.DataFrame(results[0].to("cpu").numpy().boxes.xyxy, columns=['xmin', 'ymin', 'xmax', 'ymax'])
    # Add the confidence of the prediction to the DataFrame
    predict_bbox['confidence'] = results[0].to("cpu").numpy().boxes.conf
    # Add the class of the prediction to the DataFrame
    predict_bbox['class'] = results[0].to("cpu").numpy().boxes.cls.astype(int)
    # Replace the class number with the class name from the labels_dict
    predict_bbox['name'] = predict_bbox["class"].replace(labels_dict)
    return predict_bbox


def add_bboxs_on_img(image: Image, predict: pd.DataFrame()) -> Image:
    """
    add a bounding box on the image

    Args:
    image (Image): input image
    predict (pd.DataFrame): predict from model

    Returns:
    Image: image whis bboxs
    """
    # Create an annotator object
    annotator = Annotator(np.array(image))

    # sort predict by xmin value
    predict = predict.sort_values(by=['xmin'], ascending=True)

    # iterate over the rows of predict dataframe
    for i, row in predict.iterrows():
        # create the text to be displayed on image
        text = f"{row['name']}: {int(row['confidence'] * 100)}%"
        # get the bounding box coordinates
        bbox = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
        # add the bounding box and text on the image
        annotator.box_label(bbox, text, color=colors(row['class'], True))
    # convert the annotated image to PIL image
    return Image.fromarray(annotator.result())


def detect_model(input_image: Image) -> pd.DataFrame:
    """
    Predict from sample_model.
    Base on YoloV8

    Args:
        input_image (Image): The input image.

    Returns:
        pd.DataFrame: DataFrame containing the object location.
    """
    model = object_detection_model

    predictions = model.predict(
        imgsz=640,
        source=input_image,
        conf=0.5,
        save=False,
        augment=False,
        flipud=0.0,
        fliplr=0.0,
        mosaic=0.0,
    )

    # Transform predictions to pandas dataframe
    predictions = transform_predict_to_df(predictions, model.model.names)

    return predictions

