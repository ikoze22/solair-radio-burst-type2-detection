from bs4 import BeautifulSoup
import requests
import os
from collections import defaultdict
from datetime import datetime
from flask import Flask, jsonify, send_file
from flask_cors import CORS
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
import cv2
from keras.applications.vgg16 import preprocess_input, decode_predictions

app = Flask(__name__)
CORS(app)

# Load the pre-trained VGG16 model
model = tf.keras.models.load_model(r'C:\Users\ASUS\Desktop\Newfolder\model_article')



@app.route('/')
def hello():
    return 'Hello, World!'
def getToday():
    today = datetime.today().date()
    formatted_date = today.strftime("%Y%m%d")
    return formatted_date



def resize_crop (image):
  h, w = image.shape
  if h >= 1060:
    img = cv2.resize(image,(400,300))
    cropped_image = img[15:274, 23:348]
    crop_resize_image = cv2.resize(cropped_image,(300,300))

  else:
    img = cv2.resize(image,(400,300))
    cropped_image = img[20:266, 60:350]
    crop_resize_image = cv2.resize(cropped_image,(300,300))
  return crop_resize_image




def pre_process(image):
  image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
  #cv2_imshow(image)
  kernel = np.ones((5,5),image.dtype)
  ##application d'un filtre d'ouverture
  processed_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
  #cv2_imshow(processed_image)
  ##application d'un filtre de fermeture
  processed_image = cv2.morphologyEx(processed_image, cv2.MORPH_CLOSE, kernel)
  #cv2_imshow(processed_image)
  ##application d'un filtre médian
  processed_image = cv2.medianBlur(image,5)
  #cv2_imshow(processed_image)
  ##Seuillage de l'image
  ret,th=cv2.threshold(processed_image.copy(), 15,255, cv2.THRESH_BINARY)
  #cv2_imshow(th)
  ##rechereche du contour
  contours, hierarchy = cv2.findContours(th.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
  contour_image = cv2.drawContours(image, contours, -1, (255, 255, 255), 2)
  #cv2_imshow(contour_image)
  #eleminer le contour moins d'un seuillage
  for cont in contours:
    x,y,w,h = cv2.boundingRect(cont)
    area = w*h

    if area < 500:
      cv2.fillConvexPoly(th, cont, 0)
  #cv2_imshow(th)

  processed_image = cv2.bitwise_and(processed_image, th)
  #cv2_imshow(processed_image)
  resized=resize_crop (processed_image)
  #inhance the contrast
  clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
  processed_image = clahe.apply(resized)
  #cv2_imshow(processed_image)

  return processed_image

#prediction
def predict_image(img_path):
    preprocessed_image = pre_process(img_path)
    predictions = model.predict(preprocessed_image)
    decoded_predictions = decode_predictions(predictions, top=3)[0]
    return decoded_predictions


@app.route('/images', methods=['GET'])
def get_images():
    folder = r'C:\Users\ASUS\Desktop\Newfolder\craag'
    if not os.path.exists(folder):
        os.mkdir(folder)

    images_data = defaultdict(dict)
    nb_downloaded = 0

    date = getToday()
    url = f'http://soleil.i4ds.ch/solarradio/callistoQuicklooks/?date=20230619'

    try:
        soup = BeautifulSoup(requests.get(url).text, 'html.parser')
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Error: {e}'}), 500

    for row in soup.find('table').find_all('tr'):
        try:
            info = row.find_all('td')[0].text
            row_station, row_date, row_time, _ = info.rsplit('_', maxsplit=3)
            row_hour = row_time[:2]
        except ValueError as e:
            print(f'Error: {e}')
            continue  # continue with the next row

        if row_station == "ALGERIA-CRAAG":
            try:
                img_link = 'http://soleil.i4ds.ch/solarradio/' + row.find_all('td')[2].find('a').get('href')[3:]
            except (AttributeError, TypeError, IndexError, NameError) as e:
                print(f"An error occurred: {e}")
                continue

            try:
                response = requests.get(img_link)
                response.raise_for_status()  # raise an error if the request is not successful
            except requests.exceptions.RequestException as e:
                print(f'Error: {e}')
                continue  # continue with the next row

            image_path = f'{folder}/{row_date}_{row_time}_{row_station}.png'
            with open(image_path, 'wb') as f:
                f.write(response.content)



            image = cv2.imread(image_path)
            # Preprocess the image
            image = pre_process(image)
            # Convert the image to a numpy array
            image_array = np.array(image)
            predictions = model.predict(np.expand_dims(image_array, axis=0))

            threshold = 0.5  # Adjust the threshold as needed
            predicted_labels = (predictions > threshold).astype(int)
            if predicted_labels == 1:
                result = "type II"
                print("type II")
            else:
                result = "none"
                print("none")

            # Process the image using your machine learning model
            # Add your code to apply the ML model to the image here



            # Store the image path and result in the images_data dictionary
            images_data[nb_downloaded]['path'] = image_path
            images_data[nb_downloaded]['result'] = result

            nb_downloaded += 1

    # Return the images_data dictionary as JSON response
    for key, value in images_data.items():
        value['path'] = value['path'].replace('\\\\', '\\').replace('/', '\\')
    return jsonify(images_data)


@app.route('/<path:filename>')
def serve_image(filename):
    return send_file(filename)

if __name__ == '__main__':
    app.run(debug=True)