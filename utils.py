import os

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'PNG', 'JPG', 'JPEG'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[
        1].lower() in ALLOWED_EXTENSIONS


def create_face_db(face_db_path):
    if not os.path.exists(face_db_path):
        try:
            os.makedirs(face_db_path)
        except OSError:
            raise OSError

def get_current_id(face_db_path):
    if not os.listdir(face_db_path):
        return 1
    else:
        list_subfolders_with_paths = [f.name for f in os.scandir(face_db_path) if
                                      f.is_dir()]
        list_id = [ int(str(subfolder).split('_')[0]) for subfolder in list_subfolders_with_paths]
        list_id = sorted(list_id)
        return list_id[-1]