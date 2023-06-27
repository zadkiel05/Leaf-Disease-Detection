import keras.utils as image
from keras.models import load_model
from keras.models import model_from_json
import numpy as np
import cv2

json_file = open('model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model1.h5")
print("Loaded model from disk")

cam = cv2.VideoCapture(0)
while(True):
    frame = cam.read()[1]
    img = cv2.resize(frame, (128, 128))
    test_image = img.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = loaded_model.predict(test_image)
    arr = np.array(result[0])
    print(arr)
    maxx = np.amax(arr)
    max_prob = arr.argmax(axis=0)
    max_prob = max_prob + 1
    classes = ['Healthy', 'Diseased', 'Background']
    result = classes[max_prob - 1]
    print(result)

    cv2.putText(frame, result, (10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow("Image",frame)
    key = cv2.waitKey(10)
    if key == 27:
        break
cam.release()
cv2.destroyAllWindows()
