import cv2,sklearn
from sklearn import preprocessing
from os.path import basename
import os
import numpy as np

def readImage(path):
    nimg = cv2.imread(path)
    nimg = cv2.resize(nimg,(160,160))
    nimg = nimg - 127.5
    nimg = nimg * 0.0078125
    return nimg

def convert_image(nimg):
    nimg = cv2.resize(nimg, (160, 160))
    nimg = nimg - 127.5
    nimg = nimg * 0.0078125
    return nimg

def load_faces(faces_dir,model):
    # file_emb = open("embreal0.txt", "w+")
    face_db = []
    for root, _, files in os.walk(faces_dir):
        index = 1
        for file in files:
            nimg = readImage(os.path.join(root, file))
            name = basename(root)
            input_image = np.expand_dims(nimg,axis=0)
            emb_array = model.runEmbedd(input_image)
            embedding = sklearn.preprocessing.normalize(emb_array).flatten()
            # file_emb.writelines(str(embedding).replace("\n"," ").replace("[", "").replace("]", "") +",0\n")
            face_db.append({
                "name": name + "_" + str(index),
                "feature": embedding
            })
            index = index + 1
    return face_db

def feature_compare(feature1, feature2):
    dist = np.sum(np.square(feature1- feature2))
    sim = np.dot(feature1, feature2.T)
    return dist,sim

def face_frame_embedding(face_image,facenet):
    input_images = np.zeros((1, 160, 160, 3))
    input_images[0,:] = convert_image(face_image)
    emb_array = facenet.runEmbedd(input_images)
    emb_array = sklearn.preprocessing.normalize(emb_array)
    return emb_array

def predict(face_image,facenet,face_db,VERIFICATION_THRESHOLD=0.5):
    names = []
    sims = []
    emb_array = face_frame_embedding(face_image,facenet)
    for i, embedding in enumerate(emb_array):
        embedding = embedding.flatten()
        temp_dict = {}
        for face_emb in face_db:
            _, sim = feature_compare(embedding, face_emb["feature"])
            temp_dict[face_emb["name"]] = sim
        dictResult = sorted(temp_dict.items(), key=lambda d: d[1], reverse=True)
        name = ""
        if len(dictResult) > 0 and dictResult[0][1] > VERIFICATION_THRESHOLD:
            name = dictResult[0][0]
            sim = dictResult[0][1]
        else:
            name = "unknown"
            sim = 0
        names.append(name)
        sims.append(sim)
    return names, sims