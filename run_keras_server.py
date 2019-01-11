from keras.applications import ResNet50
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from PIL import Image
import tensorflow as tf
import numpy as np
import flask
import io


# initialize Flask application, Keras model
app = flask.Flask(__name__)
graph = []
model = None


def load_model():
    # load pre-trained keras model, can be substituted
    global model
    global graph
    model = ResNet50(weights="imagenet")
    graph = tf.get_default_graph()


# pre-process image data
def prepare_image(image, target):
    """
    This function accepts and input image, converts to RGB, resizes
    to 224x224 pixels, and preprocesses the array by subtraction and
    scaling
    :param image: image to be processed
    :param target: target
    :return: processed image array
    """
    # convert image mode to RGB if not already RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize and pre-process input
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    # return processed image
    return image


# processes POST requests to the /predict endpoint
@app.route("/predict", methods=["POST"])
def predict():
    # initialize data dictionary to be returned from view
    data = {"success": False}

    # ensure image was uploaded properly to endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # pre-process image and prepare it for classification
            image = prepare_image(image, target=(224, 224))

            # classify input image and initialize list of predictions
            # to return to client
            with graph.as_default():
                preds = model.predict(image)
            results = imagenet_utils.decode_predictions(preds)
            data["predictions"] = []

            # loop results and add to list of returned predictions
            for (imagenetID, label, prob) in results[0]:
                r = {"label": label, "probability": float(prob)}
                data["predictions"].append(r)

            # indicate result was success
            data["success"] = True

    # return the data dictionary as JSON response
    return flask.jsonify(data)


# if this is man thread of execution, load model first then start server
if __name__ == "__main__":
    print(("* Loading keras model and Flask starting server..."
           "please wait until server has fully started"))
    load_model()
    app.run()
