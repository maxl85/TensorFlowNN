import os
import cv2

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np

model = VGG16(weights='imagenet')

# Start the camera
camObj = cv2.VideoCapture(1)

if not camObj.isOpened():
    print('Camera did not provide frame.')
else:
    while(True):
        # Capture frame-by-frame
        readOK, frame = camObj.read()
        
        # Display the input frame
        cv2.imshow('frame', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        x = cv2.resize(frame, (224, 224))
        x = image.img_to_array(x)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        preds = model.predict(x)
        
        os.system('cls')
        
        out = decode_predictions(preds, top=3)[0]
        for i in range(len(out)):
            print('%2d => %s' % (out[i][2]*100, out[i][1]))


# When everything done, release the capture
camObj.release()
cv2.destroyAllWindows()