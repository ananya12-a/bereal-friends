import cv2
import requests
from PIL import Image
import os
import json
import numpy as np

import matplotlib.pyplot as plt



f = open('data/Friends.txt')
f1 = open('data/Memories.txt')
memories = json.load(f1)
friends = json.load(f)['data']
f.close()
f1.close()
def saveMemories(memories):
    count=0
    for memory in memories:
        img = Image.open(requests.get(memory['primary']['url'], stream = True).raw)
        img.save(f'data/Memories/memory{count}.png')
        count+=1
def saveFriends(friends):
    for friend in friends:
        if ('profilePicture' in friend.keys() and requests.get(friend['profilePicture']['url']).status_code==200):
            # print(requests.get(friend['profilePicture']['url']).status_code)
            img = Image.open(requests.get(friend['profilePicture']['url'], stream = True).raw)
            username = friend['username']
            if (not os.path.isdir(f'data/Friends/{username}')):
                os.mkdir(f'data/Friends/{username}')
            img.save(f'data/Friends/{username}/01.png')

# saveMemories(memories)
# saveFriends(friends)
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
headshots_folder_name = 'data/Friends'
def detectFaces(headshots_folder_name):
    # dimension of images
    image_width = 2000
    image_height = 1500

    # for detecting faces
    facecascade = detector

    # set the directory containing the images
    images_dir = os.path.join(".", headshots_folder_name)

    current_id = 0
    label_ids = {}

    # iterates through all the files in each subdirectories
    for root, _, files in os.walk(images_dir):
        for file in files:
            if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
                # path of the image
                path = os.path.join(root, file)

                # get the label name (name of the person)
                label = os.path.basename(root).replace(" ", ".").lower()
                # print("label", label)

                # add the label (key) and its number (value)
                if not label in label_ids:
                    label_ids[label] = current_id
                    current_id += 1

                # load the image
                imgtest = cv2.imread(path, cv2.IMREAD_COLOR)
                print(path)
                image_array = np.array(imgtest, "uint8")

                # get the faces detected in the image
                faces = facecascade.detectMultiScale(imgtest,
                    scaleFactor=1.1, minNeighbors=5)

                # if not exactly 1 face is detected, skip this photo
                print(faces)
                if len(faces) != 1:
                    print(f'---Photo skipped---\n')
                    # remove the original image
                    os.remove(path)
                    continue

                # save the detected face(s) and associate
                # them with the label
                for (x_, y_, w, h) in faces:

                    # draw the face detected
                    face_detect = cv2.rectangle(imgtest,
                            (x_, y_),
                            (x_+w, y_+h),
                            (255, 0, 255), 2)
                    plt.imshow(face_detect)
                    plt.show()

                    # resize the detected face to 224x224
                    size = (image_width, image_height)

                    # detected face region
                    roi = image_array[y_: y_ + h, x_: x_ + w]

                    # resize the detected head to target size
                    resized_image = cv2.resize(roi, size)
                    image_array = np.array(resized_image, "uint8")

                    # remove the original image
                    os.remove(path)

                    # replace the image with only the face
                    im = Image.fromarray(image_array)
                    im.save(path)

detectFaces(headshots_folder_name)

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)] 
    # print(imagePaths)
    faceSamples=[]
    ids = []
    id = 0
    
    for imagePath in imagePaths:
        imagePath = imagePath + '/01.png'
        if (not os.path.exists(imagePath)): continue
        PIL_img = Image.open(imagePath).convert('L') #Luminance  ==> greystyle
        img_numpy = np.array(PIL_img,'uint8')
        #print(PIL_img)
        #.show()
        #print(len(img_numpy)
        id = int(os.path.split(imagePath)[-1].split(".")[0])
        faces = detector.detectMultiScale(img_numpy)
        #print(id)
        #print(faces)
        if len(faces) > 1:
            print("The following image detect more than 1 face", imagePath)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
            # id+=1
            #print(ids)
            
    return faceSamples,ids
faces,ids = getImagesAndLabels('data/Friends')
print(ids)
print("{0} faces, {0} id in total are detected".format(len(faces), len(ids)))

def train_classifier(faces, faceID):
    face_recognizer=cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces, np.array(faceID))
    return face_recognizer
# Save the model as bdktrainer.yml
facerecognizer = train_classifier(faces, ids) 
facerecognizer.save('bdktrainer.yml')

name = {}
count = 0
for friend in friends:
    if ('profilePicture' in friend.keys() and requests.get(friend['profilePicture']['url']).status_code==200):
        # print(requests.get(friend['profilePicture']['url']).status_code)
        name[count] = friend['username']
        count+=1
imgloc = 'data/Memories/memory7.png'
gray = cv2.imread(imgloc)

# gray2 = cv2.cvtColor(gray,cv2.COLOR_BGR2RGB)
#plt.imshow(gray2)
#plt.show()

grayimage2 = cv2.cvtColor(gray,cv2.COLOR_BGR2GRAY)
faces = detector.detectMultiScale(grayimage2,scaleFactor=1.2, minNeighbors = 2)

print('number of faces = {0} '.format(len(faces)))
#plt.imshow(gray)
#plt.show()


def put_text(test_img,text,x,y): 
    cv2.putText(test_img,text,(x,y),cv2.FONT_ITALIC,1.1,(255,255,255),2) #text, font size, font color( purple yellow), thickness

print(faces)

for face in faces:
    (x,y,w,h) = face
    print(x,y,w,h)
    roi_gray = grayimage2[y:y+h, x:x+h]
    plt.imshow(roi_gray, cmap='gray')
    plt.show()
    label, confidence = facerecognizer.predict(roi_gray)
    
    print("confidence:", confidence)
    print("label:", label)
    
    cv2.rectangle(gray, (x, y), (x+w, y+h), (255, 255, 255), 2)
    predicted_name = name[label]
    print(predicted_name)
    put_text(gray, predicted_name,x,y-5) #put text 5 pixel higher than face
    
    gray = cv2.cvtColor(gray,cv2.COLOR_BGR2RGB)
    
plt.imshow(gray)