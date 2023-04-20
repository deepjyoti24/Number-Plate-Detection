import cv2
import imutils
import pytesseract
import numpy as np
import pandas as pd
import sys
import time

pytesseract.pytesseract.tesseract_cmd=r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"

image = cv2.imread('image4.jpg')

image = imutils.resize(image, width= 500)
#original image shown
cv2.imshow("original image", image) 
cv2.waitKey(0)

#cover the image to gray level

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray scale image", gray)
cv2.waitKey(0)

#now reduce noise

gray = cv2.bilateralFilter(gray, 11, 17, 17)
cv2.imshow("smother", gray)
cv2.waitKey(0)

#now find thw edge of the image

edged = cv2.Canny(gray, 170, 200)
cv2.imshow("canny edged", edged)
cv2.waitKey(0)

#now we will find the contours based on the image
cnts , new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

#now we will draw all the controus

image1 = image.copy()
cv2.drawContours(image1, cnts, -1, (0,255,0), 3)
cv2.imshow("Canny after controus", image1)
cv2.waitKey(0)

cnts = sorted(cnts, key = cv2.contourArea, reverse=True)[:30]
NumberPlatecount = None

image2 = image.copy()
cv2.drawContours(image2, cnts, -1, (0,255,0), 3)
cv2.imshow("top controus", image2)
cv2.waitKey(0)

#now we will run a for loop to find the numberplate
cnt = 0
name = 1

for i in cnts:
    perimeter = cv2.arcLength(i, True)
    
    approx = cv2.approxPolyDP(i, 0.02*perimeter, True)
    
    if(len(approx) == 4):
        NumberPlatecount = approx
        x,y,w,h = cv2.boundingRect(i)
        crp_img = image[y:y+h, x:x+w]
        
        cv2.imwrite(str(name)+ '.png', crp_img)
        name += 1
        
        break
cv2.drawContours(image,[NumberPlatecount], -1, (0,255,0), 3)
cv2.imshow("final img", image)
cv2.waitKey(0)

#crop the image and convert the img to text

crop_img = '1.png'
cv2.imshow("Cropprd image", cv2.imread(crop_img))

text = pytesseract.image_to_string(crop_img, lang='eng')
print( "Number is:", text)
cv2.waitKey(0)
# for store the data in a csv file
raw_data = {'date': [time.asctime( time.localtime(time.time()) )],'v_number': [text]}

df = pd.DataFrame(raw_data, columns = ['date','v_number'])
df.to_csv('data.csv')

    

        
    
