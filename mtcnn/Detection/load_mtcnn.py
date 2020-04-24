from mtcnn.Detection.MtcnnDetector import MtcnnDetector
from mtcnn.Detection.detector import Detector
from mtcnn.Detection.fcn_detector import FcnDetector
from mtcnn.nets.mtcnn_model import P_Net, R_Net, O_Net
import configparser
def load_mtcnn(scale_factor):
    conf = configparser.ConfigParser()
    conf.read("mtcnn/config/main.cfg")
    # load mtcnn model
    MODEL_PATH = conf.get("MTCNN", "MODEL_PATH")
    MIN_FACE_SIZE = int(conf.get("MTCNN", "MIN_FACE_SIZE"))
    STEPS_THRESHOLD = [float(i)  for i in conf.get("MTCNN", "STEPS_THRESHOLD").split(",")]

    detectors = [None, None, None]
    prefix = [MODEL_PATH + "/PNet_landmark/PNet",
              MODEL_PATH + "/RNet_landmark/RNet",
              MODEL_PATH + "/ONet_landmark/ONet"]
    epoch = [18, 14, 16]
    model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
    PNet = FcnDetector(P_Net, model_path[0])
    detectors[0] = PNet
    RNet = Detector(R_Net, 24, 1, model_path[1])
    detectors[1] = RNet
    ONet = Detector(O_Net, 48, 1, model_path[2])
    detectors[2] = ONet
    mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=MIN_FACE_SIZE, threshold=STEPS_THRESHOLD, scale_factor=scale_factor)
    return mtcnn_detector