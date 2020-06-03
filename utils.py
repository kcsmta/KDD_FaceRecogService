import configparser
import os
from mtcnn.Detection.load_mtcnn import load_mtcnn
from face_encoder.facenet import Facenet
from face_encoder.support import load_faces

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'PNG', 'JPG', 'JPEG'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[
        1].lower() in ALLOWED_EXTENSIONS


def get_current_id(face_db_name):
    if len(face_db_name) == 0:
        return 0
    ID = int(face_db_name[-1].split("_")[0]) + 1
    return ID


def init_recognizer():
    mtcnn_detector = load_mtcnn(scale_factor=0.709)
    facenet = Facenet("face_encoder/models/20180402-114759.pb")
    conf = configparser.ConfigParser()
    conf.read("config/global.cfg")
    FACE_DB_PATH = conf.get("PATH", "FACE_DB_PATH")
    face_db, face_db_name = load_faces(FACE_DB_PATH, facenet)
    return mtcnn_detector, facenet, face_db, face_db_name
