from keras.models import load_model
from tkinter import *
import os
import glob
import cv2
from PIL import ImageGrab, Image , ImageDraw
import numpy as np

model = load_model(r'/Users/manaalsaxena/Downloads/ML Projects/handwritingg/mnist.h5')

#create a main window
root = Tk()
root.resizable(0,0)
root.title("Handwritten Digit Recognizer Gui App")

lastx, lasty = None, None
image_number = 0

#clear window function
def clear_widget():
    global cv
    # to clear canvas
    cv.delete("all")

def activate_event(event):
    global lastx, lasty
    cv.bind('<B1-Motion>', draw_lines)
    lastx, lasty = event.x, event.y

#draw lines
def draw_lines(event):
    global lastx, lasty
    x, y = event.x, event.y
    cv.create_line((lastx, lasty, x, y), width=8, fill='black', capstyle=ROUND, smooth=True, splinesteps=12)
    lastx, lasty = x, y

#recognize digit
def Recognize_Digit():
    global image_number
    predictions = []
    prcentage = []
    filename = f"image_{image_number}.png"
    widget = cv

    x = root.winfo_rootx()+widget.winfo_x()
    y = root.winfo_rooty()+widget.winfo_y()
    x1 = x+widget.winfo_width()
    y1 = y+widget.winfo_height()

    ImageGrab.grab().crop((x,y,x1,y1)).save(filename)

#create a canvas
cv = Canvas(root, height=480, width=680, bg='black')
cv.grid(row=0,column=0,pady=2,sticky=W , columnspan=2)
cv.bind('<Button-1>', activate_event)

#adding buttons and labels
btn_save = Button(text="Recognize Digit", command=Recognize_Digit)
btn_save.grid(row=2, column=0, pady=1, padx=1)
button_clear = Button(text="Clear Widget", command= clear_widget)
button_clear.grid(row=2,column=1,padx=1,pady=1)

#run this when application is ready
root.mainloop()


# read the image in color format
filename = f"image_{image_number}.png"
image = cv2. imread(filename, cv2. IMREAD_COLOR)
# convert the image in grayscale
gray = cv2. cvtColor(image, cv2.COLOR_BGR2GRAY)
# applying Otsu thresholding
ret, th = cv2.threshold(gray, 0, 255,cv2. THRESH_BINARY_INV+cv2.THRESH_OTSU)
#findcontour () function helps in extracting the contours from the image.
contours= cv2.findContours(th, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]

for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    #rectangle
    cv2.rectangle(image,(x,y),(x+w, y+h),(255,0,0),1)
    top = int(0.05 * th.shape[0])
    bottom = top
    left = int(0.05 * th.shape[1])
    right = left
    th_up = cv2.copyMakeBorder(th, top, bottom, left, right, cv2.BORDER_REPLICATE)
    # Extract the image ROI
    roi = th[y - top: y + h + bottom, x - left: x+w+right]
    # resize roi image to 28x28 pixels
    img = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    # reshaping the image to support our model input
    img = img.reshape(1, 28, 28, 1)
    # normalizing the image to support our model input
    img = img / 255.0
    # its time to predict the result
    pred = model.predict([img])[0]
    # numpy. argmax (input array) Returns the indices of the maximum values,
    final_pred = np.argmax(pred)
    data = str(final_pred)+' ' + str(int(max(pred)*100))+'%'
    # cv2.putText () method is used to draw a text string on image.
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    color = (255, 0, 0)
    thickness = 1
    cv2.putText(image, data, (x, y - 5), font, fontScale , color, thickness)

# Showing the predicted results on new window.z
cv2.imshow('image', image)
cv2.waitKey(0)