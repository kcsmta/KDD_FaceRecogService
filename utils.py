import configparser
import os
from mtcnn.Detection.load_mtcnn import  load_mtcnn
from face_encoder.facenet import Facenet
from  face_encoder.support import load_faces
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'PNG', 'JPG', 'JPEG'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[
        1].lower() in ALLOWED_EXTENSIONS


def create_folder(face_db_path, predict_path, temp_path):
    if not os.path.exists(face_db_path):
        try:
            os.makedirs(face_db_path)
        except OSError:
            raise OSError
    if not os.path.exists(predict_path):
        try:
            os.makedirs(predict_path)
        except OSError:
            raise OSError
    if not os.path.exists(temp_path):
        try:
            os.makedirs(temp_path)
        except OSError:
            raise OSError
def get_current_id(face_db_path):
    if not os.listdir(face_db_path):
        return 1,[]
    else:
        list_subfolders_with_paths = [f.name for f in os.scandir(face_db_path) if
                                      f.is_dir()]
        list_id = [ int(str(subfolder).split('_')[0]) for subfolder in list_subfolders_with_paths]
        list_id = sorted(list_id)
        return list_id[-1]+1, list_id
def init_recognizer():
    mtcnn_detector = load_mtcnn(scale_factor=0.709)
    facenet = Facenet("face_encoder/models/20180402-114759.pb")
    conf = configparser.ConfigParser()
    conf.read("config/global.cfg")
    FACE_DB_PATH = conf.get("PATH", "FACE_DB_PATH")
    face_db,face_db_name = load_faces(FACE_DB_PATH, facenet)
    return mtcnn_detector, facenet, face_db,face_db_name
