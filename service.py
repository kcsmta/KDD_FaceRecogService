import shutil
import cv2
import flask
from flask import request, jsonify
from werkzeug.utils import secure_filename
from face_encoder.support import  predict
from utils import *
from waitress import serve

face_db_path = 'face_db/'
predict_path = 'predict/'
temp_path = 'temp/'
ID = None
list_id = None
mtcnn_detector, facenet, face_db = None, None, None

app = flask.Flask(__name__, static_url_path='', static_folder='')
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # max 200mb in a request
app.config["DEBUG"] = True  # for debug
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'


@app.route('/add_person', methods=['POST'])
def add_person():
    global ID, list_id
    if request.method == 'POST':
        files = request.files.getlist('file')
        if request.files['file'].filename=='':
            return jsonify(message='No file selected for uploading'), 400
        elif 'name' not in request.form:
            return jsonify(message='Bad request! No name exist!'), 400
        elif request.form['name']=='':
            return jsonify(message='Bad request! Name is empty!'), 400
        else:
            name = request.form['name']
            folder_new_person = os.path.join(face_db_path, str(ID) + '_' + name)
            ID = ID + 1
            os.makedirs(folder_new_person)
            for file in files:
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    path_to_file = os.path.join(
                            folder_new_person, filename)
                    print(path_to_file)
                    file.save(path_to_file)
            list_id.append(ID)
            return jsonify(message='upload success'), 200


@app.route('/del_person', methods=['GET'])
def del_person():
    if 'id_person' not in request.form:
        return jsonify(message='Bad request! No id_person exist!'), 400
    else:
        id_person = request.form['id_person']
        global list_id
        if not int(id_person):
            return jsonify(message='id_person must be integer'), 400
        if int(id_person) in list_id:
            for folder in os.scandir(face_db_path):
                if int(str(folder.name).split('_')[0]) == int(id_person):
                    shutil.rmtree(folder)
                    return jsonify(message='delete on id_person ' + id_person), 200
        else:
            return jsonify(message='id does not exist'), 400


@app.route('/predict', methods=['POST'])
def predict_img():
    global mtcnn_detector, facenet, face_db
    if request.method == 'POST':
        files = request.files.getlist('file')
        # print(request.files['file'].filename)
        # thresh = request.form['thresh']
        if request.files['file'].filename=='':
            return jsonify(message='No file selected for predict'), 400
        else:
            result = {}
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
            return jsonify(str(result)), 200
if __name__ == '__main__':
    create_folder(face_db_path, predict_path, temp_path)
    mtcnn_detector, facenet, face_db = init_recognizer()
    ID, list_id = get_current_id(face_db_path)
    # app.run(debug=app.config['DEBUG'], use_reloader=False)
    serve(app, host='0.0.0.0', port=8000)