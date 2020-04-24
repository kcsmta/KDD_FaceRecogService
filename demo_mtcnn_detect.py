from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
from mtcnn.Detection.load_mtcnn import  load_mtcnn
def main():
    cam = cv2.VideoCapture(0)
    mtcnn_detector = load_mtcnn(scale_factor=0.709)
    while True:
        ret,frame = cam.read()
        if ret:
            faces,_ = mtcnn_detector.detect(frame)
            for face in faces:
                x1,y1,x2,y2 = int(face[0]),int(face[1]),int(face[2]),int(face[3])
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),thickness=2)
            cv2.imshow("frame",frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            if key == 32:
                cv2.waitKey(0)
        else:
            break
if __name__ == '__main__':
    main()