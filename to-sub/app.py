from flask import Flask, request, redirect, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import sys
from pathlib import Path
if __package__ is None:                  
    DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(DIR.parent))
    __package__ = DIR.name
from .media import *
from .log import logger, init_logger
from tensorflow.keras.models import load_model
from flask_cors import CORS, cross_origin

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'mp4', 'mkv', 'avi'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CORS_HEADERS'] = 'Content-Type'

cors = CORS(app, resources={r"/api/upload": {"origins": "http://localhost:port"}})

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/upload', methods=['GET', 'POST'])
@cross_origin(origin='*',headers=['Content- Type','Authorization'])
def run():
  if request.method == 'POST':
    if 'file' not in request.files:
        print('No file part')
        return redirect(request.url)
    file = request.files['file']
    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
        print('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('download_file', name=filename))

  files = []
  media = [Media(m) for m in files]
  model = load_model('model/out/ann.hdf5')
  model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

  for m in media:
    m.mfcc()
  for s in m.subtitles():
    s.determine_speech(model)
    s.to_srt()