from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from mtcnn.Detection.load_mtcnn import  load_mtcnn
## facenet
import cv2
from face_encoder.facenet import Facenet
from  face_encoder.support import load_faces,predict
import configparser

def test_with_cam(facenet,facedb):
    cam = cv2.VideoCapture(0)
    mtcnn_detector = load_mtcnn(scale_factor=0.709)
    while True:
        ret, frame = cam.read()
        if ret:
            faces, _ = mtcnn_detector.detect(frame)
            for face in faces:
                x1, y1, x2, y2 = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                face_image = frame[y1:y2,x1:x2]
                names, sims = predict(face_image, facenet, facedb, VERIFICATION_THRESHOLD=0.5)
                print(names)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
            cv2.imshow("frame", frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            if key == 32:
                cv2.waitKey(0)
        else:
            break

def predict_image(image):
    names, sims = predict(image, facenet, face_db, VERIFICATION_THRESHOLD=0.5)
    print(names)

if __name__ == '__main__':
    # load model
    facenet = Facenet("face_encoder/models/20180402-114759.pb")
    conf = configparser.ConfigParser()
    conf.read("config/global.cfg")
    FACE_DB_PATH = conf.get("PATH", "FACE_DB_PATH")
    face_db = load_faces(FACE_DB_PATH,facenet)

    print(len(face_db))
    test_with_cam(facenet,face_db)
    # image = cv2.imread("face_db/dan/0.jpg")
    # predict_image(image)
#