import shutil
import time

import cv2
import flask
from flask import request, jsonify
from werkzeug.utils import secure_filename
from face_encoder.support import predict,face_frame_embedding
from utils import *
from waitress import serve
import numpy as np

face_db_path = 'face_db/'
predict_path = 'predict/'
temp_path = 'temp/'
ID = None
list_id = None
mtcnn_detector, facenet, face_db, face_db_name = None, None, None,None

app = flask.Flask(__name__, static_url_path='', static_folder='')
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # max 200mb in a request
app.config["DEBUG"] = True  # for debug
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

@app.route('/get_all', methods=['GET'])
def get_all():
    global face_db_path,face_db_name
    list_subfolders_with_paths = [f.name for f in os.scandir(face_db_path) if
                                  f.is_dir()]
    result = {}
    for index,element in enumerate(face_db_name):
        result[""+str(index)] = element
    return jsonify(result)
    for subfolder in list_subfolders_with_paths:
        result[str(subfolder).split('_')[0]] = str(subfolder).split('_')[1]
    return jsonify(result)

@app.route('/add_person', methods=['POST'])
def add_person():
    global ID, list_id,face_db,face_db_name
    if request.method == 'POST':
        files = request.files.getlist('file')
        if request.files['file'].filename == '':
            return jsonify(message='No file selected for uploading'), 400
        elif 'name' not in request.form:
            return jsonify(message='Bad request! No name exist!'), 400
        elif request.form['name'] == '':
            return jsonify(message='Bad request! Name is empty!'), 400
        else:
            name = request.form['name']
            folder_new_person = os.path.join(face_db_path, str(ID) + '_' + name)
            save_name = str(ID) + '_' + name
            ID = ID + 1
            os.makedirs(folder_new_person)
            result = {}
            detail = {}
            skip = True
            for file in files:
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    path_to_file = os.path.join(
                            folder_new_person, filename)
                    print(path_to_file)
                    path_temp = os.path.join(
                            temp_path, filename)
                    if os.path.isfile(path_temp):
                        os.remove(path_temp)
                    file.save(path_temp)
                    print(request.files['file'].read())
                    img = cv2.imread(path_temp)
                    faces, _ = mtcnn_detector.detect(img)
                    if len(faces) != 1:
                        detail[file.filename] = 'Image must have only 1 face!'
                        # os.remove(path_temp)
                    else:
                        x1, y1, x2, y2 = int(faces[0][0]), int(
                                faces[0][1]), int(
                                faces[0][2]), int(faces[0][3])
                        # print('face size ', y2-y1, x2-x1)
                        if y2 - y1 < 161 or x2 - x1 < 161:
                            detail[file.filename] = 'Need bigger face in image!'
                            # os.remove(path_temp)
                        ### save and add embedding to numppy array here
                        else:
                            face_image = img[y1:y2, x1:x2]
                            embedding = face_frame_embedding(face_image, facenet)
                            face_db = np.append(face_db,{
                                "name":save_name,
                                "feature": embedding
                            })
                            np.save('face_db_embed.npy',face_db)
                            skip = False
                            cv2.imwrite(path_to_file, face_image)
                            detail[file.filename] = 'Success'
                    os.remove(path_temp)
            if not skip:
                face_db_name = np.append(face_db_name, save_name)
                np.save('face_db_name.npy', face_db_name)
            if len(os.listdir(folder_new_person)) != 0:
                list_id.append(ID)
                result = {'status':'success','id':str(ID),'detail':detail}
            else:
                ID = ID -1
                os.rmdir(folder_new_person)
                result = {'status': 'fail', 'id': str(-1), 'detail': detail}
            return jsonify(result), 200


@app.route('/del_person', methods=['POST'])
def del_person():
    global  face_db,face_db_name
    if 'id_person' not in request.form:
        return jsonify(message='Bad request! No id_person exist!'), 400
    else:
        id_person = request.form['id_person']
        global list_id
        if not int(id_person):
            return jsonify(message='id_person must be integer'), 400
        ## delete in numpy array and save
        for index, element in enumerate(face_db):
            if element['name'].split('_')[0] == id_person:
                face_db = np.delete(face_db, index)
        for index, element in enumerate(face_db_name):
            if element.split('_')[0] == id_person:
                face_db_name = np.delete(face_db_name, index)
        np.save('face_db_name.npy', face_db_name)
        np.save('face_db_embed.npy',face_db)
        if int(id_person) in list_id:
            for folder in os.scandir(face_db_path):
                if int(str(folder.name).split('_')[0]) == int(id_person):
                    shutil.rmtree(folder)
                    return jsonify(
                        message='delete on id_person ' + id_person), 200
        else:
            return jsonify(message='id does not exist'), 400


@app.route('/predict', methods=['POST'])
def predict_img():
    global mtcnn_detector, facenet, face_db
    if request.method == 'POST':
        files = request.files.getlist('file')
        # print(request.files['file'].filename)
        # thresh = request.form['thresh']
        if request.files['file'].filename == '':
            return jsonify(message='No file selected for predict'), 400
        else:
            result = {}
            start = time.time()
            for file in files:
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    path_to_file = os.path.join(
                            predict_path, filename)
                    print(path_to_file)
                    if os.path.isfile(path_to_file):
                        os.remove(path_to_file)
                    file.save(path_to_file)
                    img = cv2.imread(path_to_file)
                    faces, _ = mtcnn_detector.detect(img)
                    names, sims = [], []
                    for face in faces:
                        x1, y1, x2, y2 = int(face[0]), int(face[1]), int(
                                face[2]), int(face[3])
                        face_image = img[y1:y2, x1:x2]
                        names, sims = predict(face_image, facenet, face_db,
                                              VERIFICATION_THRESHOLD=0.5)
                    result[filename] = len(names), names, sims
            end = time.time()
            run_time = end-start
            time_sleep = 0.5*len(files)- run_time
            print(run_time,time_sleep)
            if time_sleep>0:
                time.sleep(time_sleep)
            return jsonify(str(result)), 200


if __name__ == '__main__':
    create_folder(face_db_path, predict_path, temp_path)
    mtcnn_detector, facenet, face_db,face_db_name = init_recognizer()
    ID, list_id = get_current_id(face_db_path)
    # app.run(debug=app.config['DEBUG'], use_reloader=False)
    # app.run(debug=True)
    # serve(app, host='0.0.0.0', port=8000)
    serve(app, host='localhost', port=8000)