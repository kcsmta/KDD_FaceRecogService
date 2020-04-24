import os
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



@app.route('/add_person/<name>', methods=['POST'])
def add_person(name):
    if request.method == 'POST':
        files = request.files.getlist('file')
        if files.count == 0:
            return jsonify(status='fail', message='No file selected for uploading')
        else:
            for file in files:
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    path_to_file = os.path.join(
                            face_db_path, filename)
                    file.save(path_to_file)
            return jsonify(status='success', message='upload success')
if __name__ == '__main__':
    create_face_db(face_db_path)
    init_recognizer()
    ID = get_current_id(face_db_path)
    print(ID)
    app.run(debug=app.config['DEBUG'])
