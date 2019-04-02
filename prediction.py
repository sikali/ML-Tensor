import cv2 
import tensorflow as tf
import matplotlib.pyplot as plt
import os



CATEGORIES = ["Dog","Cat"]

def prepare(filepath):
    IMG_SIZE = 50
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

model = tf.keras.models.load_model("MLModel")

for x  in range(9):
    
    print(x)
    num = str(x)
    pic = '/Users/alexali/Desktop/pets/'+ num + '.jpg'
    prediction = model.predict([prepare(pic)])

    

    print(CATEGORIES[int(math.ceil(prediction[0][0]))])
    print(math.ceil(prediction))
    img_array = cv2.imread(pic, cv2.IMREAD_GRAYSCALE) #Keep Grayscale because if its in Color it will be multi-dimensional - needs to normalized
    plt.imshow(img_array,cmap="gray")
    plt.show()
    plt.show(pic)
