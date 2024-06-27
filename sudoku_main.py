import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os, random
import cv2
from glob import glob
import sklearn
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import ops
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from keras.preprocessing.image import ImageDataGenerator, load_img
# from keras.utils.np_utils import to_categorical
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Activation, Dropout, Dense, Flatten, BatchNormalization, Conv2D, MaxPooling2D
# from tensorflow.keras.optimizers import RMSprop
# from tensorflow.keras import backend as K
# from tensorflow.keras.preprocessing import image
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path
from PIL import Image

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

#Loading the data
data = os.listdir("digits updated/digits updated")
data_x = []
data_y = []
data_classes = len(data)
for i in range (0, data_classes):
    data_list = os.listdir("digits updated/digits updated" + "/" + str(i))
    for j in data_list:
        pic = cv2.imread("digits updated/digits updated" + "/" + str(i) + "/" + j)
        pic = cv2.resize(pic, (32, 32))
        data_x.append(pic)
        data_y.append(i)
        # if len(data_x) == len(data_y):
        #     print("Total Datapoints = ", len(data_x))

#Labels and images
data_x = np.array(data_x)
data_y = np.array(data_y)

#Splitting the train validation and train sets
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.05)
train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.2)
# print("Training Set Shape = ", train_x.shape)
# print("Validation Set Shape = ", valid_x.shape)
# print("Test Set Shape = ", test_x.shape)

#Pre-processing the images for Neural Net
def Prep(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #making image grayscale
    img = cv2.equalizeHist(img) #Histogram Equalization to enhance contrast
    img = img/255 #Normalizing
    return img

train_x = np.array(list(map(Prep, train_x)))
test_x = np.array(list(map(Prep, test_x)))
valid_x = np.array(list(map(Prep, valid_x)))

#Reshaping the images
train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], train_x.shape[2], 1)
test_x = test_x.reshape(test_x.shape[0], test_x.shape[1], test_x.shape[2], 1)
valid_x = valid_x.reshape(valid_x.shape[0], valid_x.shape[1], valid_x.shape[2], 1)

#Augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, shear_range=0)
datagen.fit(train_x)

#OneHotEncoding
train_y = keras.utils.to_categorical(train_y, data_classes)
test_y = keras.utils.to_categorical(test_y, data_classes)
valid_y = keras.utils.to_categorical(valid_y, data_classes)

#Model Building
model = keras.Sequential()
model.add((layers.Conv2D(60, (5,5), input_shape=(32,32,1), padding="same", activation="relu")))
model.add((layers.Conv2D(60, (5,5), padding="same", activation="relu")))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
#model.add(layers.Dropout(0.25))
model.add((layers.Conv2D(30, (3,3), padding="same", activation="relu")))
model.add((layers.Conv2D(30, (3,3), padding="same", activation="relu")))
model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(layers.Dropout(0.5))
model.add(layers.Flatten())
model.add(layers.Dense(500, activation="relu"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation="softmax"))
model.summary()

#Compiling the model
optimizer = keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-08, weight_decay=0.0)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

#Fit the model
history = model.fit(datagen.flow(train_x, train_y, batch_size=32),
                    epochs=3, validation_data=(valid_x, valid_y),
                    verbose=2, steps_per_epoch=200)

#Test the model on the test set
score = model.evaluate(test_x, test_y, verbose=0)
# print("Test Score = ", score[0])
# print("Test Accuracy = ", score[1])

#Randomly select an image from dataset
folder = "sudoku-box-detection/aug"
a = random.choice(os.listdir(folder))
print(a)
sudoku_a = cv2.imread(folder+'/'+a)
# plt.figure()
# plt.imshow(sudoku_a)
# plt.show()

#Pre-processing image to be read
sudoku_a = cv2.resize(sudoku_a, (450, 450))

#Function to grayscale, blur and change the receptive threshold
def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 6)
    #blur = cv2.bilateralFilter(gray, 9, 75, 75)
    threshold_img = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    return threshold_img

threshold = preprocess(sudoku_a)

#Let's have a look at what we've got
# plt.figure()
# plt.imshow(threshold)
# plt.show()

#Finding the outline of sudoku puzzle in the image
contour_1 = sudoku_a.copy()
contour_2 = sudoku_a.copy()
contour, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(contour_1, contour, -1, (0, 255, 0), 3)

#Let's see what we got
# plt.figure()
# plt.imshow(contour_1)
# plt.show()

def main_outline(contour):
    biggest = np.array([])
    max_area = 0
    for i in contour:
        area = cv2.contourArea(i)
        if area >50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i , 0.02* peri, True)
            if area > max_area and len(approx) ==4:
                biggest = approx
                max_area = area
    return biggest ,max_area
def reframe(points):
    points = points.reshape((4, 2))
    points_new = np.zeros((4,1,2),dtype = np.int32)
    add = points.sum(1)
    points_new[0] = points[np.argmin(add)]
    points_new[3] = points[np.argmax(add)]
    diff = np.diff(points, axis =1)
    points_new[1] = points[np.argmin(diff)]
    points_new[2] = points[np.argmax(diff)]
    return points_new
def splitcells(img):
    rows = np.vsplit(img,9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r,9)
        for box in cols:
            boxes.append(box)
    return boxes
black_img = np.zeros((450,450,3), np.uint8)
biggest, maxArea = main_outline(contour)
if biggest.size != 0:
    biggest = reframe(biggest)
    cv2.drawContours(contour_2,biggest,-1, (0,255,0),10)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0,0],[450,0],[0,450],[450,450]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imagewrap = cv2.warpPerspective(sudoku_a,matrix,(450,450))
    imagewrap =cv2.cvtColor(imagewrap, cv2.COLOR_BGR2GRAY)

# plt.figure()
# plt.imshow(imagewrap)
# plt.show()

# Importing puzzle to be solved
puzzle = cv2.imread("su-puzzle/su5.jpg")
#let's see what we got
# plt.figure()
# plt.imshow(puzzle)
# plt.show()

# Preprocessing Puzzle 
su_puzzle = preprocess(puzzle)

# Finding the outline of the sudoku puzzle in the image
su_contour_1= su_puzzle.copy()
su_contour_2= sudoku_a.copy()
su_contour, hierarchy = cv2.findContours(su_puzzle,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(su_contour_1, su_contour,-1,(0,255,0),3)
black_img = np.zeros((450,450,3), np.uint8)
su_biggest, su_maxArea = main_outline(su_contour)
if su_biggest.size != 0:
    su_biggest = reframe(su_biggest)
cv2.drawContours(su_contour_2,su_biggest,-1, (0,255,0),10)
su_pts1 = np.float32(su_biggest)
su_pts2 = np.float32([[0,0],[450,0],[0,450],[450,450]])
su_matrix = cv2.getPerspectiveTransform(su_pts1,su_pts2)  
su_imagewrap = cv2.warpPerspective(puzzle,su_matrix,(450,450))
su_imagewrap =cv2.cvtColor(su_imagewrap, cv2.COLOR_BGR2GRAY)
# plt.figure()
# plt.imshow(su_imagewrap)
# plt.show()

#Splitting the cells and Classifying digits
sudoku_cell = splitcells(su_imagewrap)
#Let's have alook at the last cell
# plt.figure()
# plt.imshow(sudoku_cell[58])
# plt.show()

def CropCell(cells):
    Cells_croped = []
    for image in cells:
        img = np.array(image)
        img = img[4:46, 6:46]
        img = Image.fromarray(img)
        Cells_croped.append(img)
    return Cells_croped
sudoku_cell_croped= CropCell(sudoku_cell)

#Let's have alook at the last cell
# plt.figure()
# plt.imshow(sudoku_cell_croped[58])
# plt.show()

def read_cells(cell,model):

    result = []
    for image in cell:
        # Preprocess the image as it was in the model 
        img = np.asarray(image)
        img = img[4:img.shape[0] - 4, 4:img.shape[1] -4]
        img = cv2.resize(img, (32, 32))
        img = img / 255
        img = img.reshape(1, 32, 32, 1)
        # Getting predictions and setting the values if probabilities are above 65% 
        
        predictions = model.predict(img)
        classIndex = np.argmax(predictions)
        probabilityValue = np.max(predictions)
        
        if probabilityValue > 0.65:
            result.append(int(classIndex))
        else:
            result.append(0)
    return result

grid = read_cells(sudoku_cell_croped, model)
grid = np.asarray(grid)

# Reshaping the grid to a 9x9 matrix
grid = np.reshape(grid,(9,9))
print(grid)

plt.figure()
plt.imshow(su_imagewrap)
plt.show()

#Solving the puzzle
def next_box(quiz):
    for row in range(9):
        for col in range(9):
            if quiz[row][col] == 0:
                return (row, col)
    return False

#Function to fill in the possible values by evaluating rows collumns and smaller cells

def possible (quiz,row, col, n):
    #global quiz
    for i in range (0,9):
        if quiz[row][i] == n: #and row != i:
            return False
    for i in range (0,9):
        if quiz[i][col] == n: #and col != i:
            return False
        
    row0 = row - row % 3 #(row)//3
    col0 = col - col % 3 #(col)//3
    for i in range(3): #(row0*3, row0*3 + 3):
        for j in range(3): #(col0*3, col0*3 + 3):
            if quiz[i+row0][j+col0]==n: #and (i,j) != (row, col):
                return False
    return True

#Recursion function to loop over untill a valid answer is found. 

# def solve(quiz):
#     val = next_box(quiz)
#     if val is False:
#         return True
#     else:
#         row, col = val
#         for n in range(1,10): #n is the possible solution
#             if possible(quiz,row, col, n):
#                 quiz[row][col]=n
#                 if solve(quiz):
#                     return True 
#                 else:
#                     quiz[row][col]=0
#         return 

#Recursion function to loop over untill a valid answer is found. 
def solve(grid, row, col):
    if (row == 9 - 1 and col == 9):
        return True
    if col == 9:
        row += 1
        col = 0
    if grid[row][col] > 0:
        return solve(grid, row, col + 1)
    for n in range(1, 9 + 1, 1):
        if possible(grid, row, col, n):
            grid[row][col] = n
            if solve(grid, row, col + 1):
                return True
        grid[row][col] = 0
    return False
def Solved(quiz):
    for row in range(9):
        if row % 3 == 0 and row != 0:
            print("....................")

        for col in range(9):
            if col % 3 == 0 and col != 0:
                print("|", end=" ")

            if col == 8:
                print(quiz[row][col])
            else:
                print(str(quiz[row][col]) + " ", end="")

solve(grid,0,0)
if solve(grid,0,0):
    Solved(grid)
else:
    print("Solution don't exist. Model misread digits.")