import os
import shutil

import flask
import requests
from flask import flash, request, redirect, render_template, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from flask_restful import Resource, Api
from utils import *
face_db_path = 'face_db/'

app = flask.Flask(__name__, static_url_path='', static_folder='face_db/')
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # max 200mb in a request
app.config["DEBUG"] = True  # for debug
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

ID = None
list_id =None

@app.route('/add_person/<name>', methods=['POST'])
def add_person(name):
    global ID, list_id
    if request.method == 'POST':
        files = request.files.getlist('file')
        assert files, "Where's my file?"
        if files.count == 0:
            return jsonify(status='fail', message='No file selected for uploading')
        else:

            folder_new_person = os.path.join(face_db_path,str(ID)+'_'+name)
            ID = ID +1
            os.makedirs(folder_new_person)
            for file in files:
                print(file)
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    path_to_file = os.path.join(
                            folder_new_person, filename)
                    print(path_to_file)
                    file.save(path_to_file)
            list_id.append(ID)
            return jsonify(status='success', message='upload success')
@app.route('/del_person/<id_person>', methods=['GET'])
def del_person(id_person):
    if not int(id_person):
        return jsonify(status='failed',message='id_person must be integer')
    if int(id_person) in list_id:
        for folder in os.scandir(face_db_path):
            if int(str(folder.name).split('_')[0]) == int(id_person):
                shutil.rmtree(folder)
                return jsonify(status='success', message='delete on id_person ' + id_person)
    else:
        return jsonify(status='fail', message='id does not exist')

if __name__ == '__main__':
    create_face_db(face_db_path)
    init_recognizer()
    ID, list_id = get_current_id(face_db_path)
    print(ID)
    app.run(debug=app.config['DEBUG'])
