import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
classes=["Covid19","Normal","pneumonia"]
ml=load_model("D:\Demo\Data\covid_pneu_model.h5")
img=image.load_img("D:\\Demo\\Data\\test\\COVID19\\COVID19(463).jpg",target_size=(224,224))
img=image.img_to_array(img)
img=np.expand_dims(img,axis=0)/255.0

prd=ml.predict(img)
ind=np.argmax(prd[0])
print(classes[ind])