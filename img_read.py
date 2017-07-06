# load and resize image data
import numpy as np
from PIL import Image
train_x = np.zeros((10000, 154587), dtype=np.float32)
test_x = np.zeros((1000, 154587), dtype=np.float32)
train_y = np.zeros(10000, dtype=np.int32)
test_y = np.zeros(1000, dtype=np.int32)

# file directory
dir = 'G:/cnn/data10/train/'
for i in range(10):
    for j in range(1000):
        file = dir + str(i+1) + '_' + str(j+1) + '.jpg'
        print(file)
        # read image
        img = Image.open(file)
        # convert image to np.array
        img_array = np.array(img, dtype=np.float32)
        img_array = img_array / 255
        img_array = img_array.reshape(1,154587)
        train_x[j*10+i,:] = img_array
        train_y[j*10+i] = i
    for j in range(100):
        file = 'G:/cnn/data10/test/' + str(i+1) + '_' + str(j+1) + '.jpg'
        print(file)
        img = Image.open(file)
        img_array = np.array(img, dtype=np.float32)
        img_array = img_array / 255
        img_array = img_array.reshape(1, 154587)
        test_x[j*10+i, :] = img_array
        test_y[j*10+i] = i

np.save("train_x_10.npy", train_x)
np.save("test_x_10.npy", test_x)
np.save("train_y_10.npy", train_y)
np.save("test_y_10.npy", test_y)
