# import necessary packages
import requests

# initialize Keras REST API endpoint URL along with input image path
KERAS_REST_API_URL = "http://localhost:5000/predict"
IMAGE_PATH = "tree.jpeg"

# load input image and construct payload for request
image = open(IMAGE_PATH, "rb").read()
payload = {"image": image}

# submit request
r = requests.post(KERAS_REST_API_URL, files=payload).json()

# check if request was successful
if r["success"]:
    # loop over predictions and display
    for (k, result) in enumerate(r["predictions"]):
        print("{}. {}: {:.4f}".format(k + 1, result["label"],
                                      result["probability"]))

# if request fails
else:
    print("Request failed")
