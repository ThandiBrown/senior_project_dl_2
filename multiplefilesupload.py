import shutil
import os
from flask_ngrok import run_with_ngrok
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename

import tensorflow as tf
import numpy as np

app = Flask(__name__)
#run_with_ngrok(app)

UPLOAD_FOLDER = "static/user/"
ALLOWED_FILE_TYPE = set(['png', 'jpg', 'jpeg'])
class_names = ['handwritten', 'machine']

new_model = tf.keras.models.load_model('model/hm_model7')

def produce_predictions(files_list):
    ''' Placeholder '''

    handwritten = []
    machine = []
    unknown = []
    original_file_list = []
            
    base_path = UPLOAD_FOLDER

    for file in files_list:
        pathname = base_path + file.filename

        img_height = 1000
        img_width = 750
        img = tf.keras.utils.load_img(
            pathname, target_size=(img_height, img_width)
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        predictions = new_model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        class_prediction = class_names[np.argmax(score)]
        percent_confidence = 100 * np.max(score)

        # if confidence percent is this low make it unknown
        if percent_confidence < 80:
            unknown.append(file.filename)
        elif class_prediction == "handwritten":
            handwritten.append(file.filename)
        elif class_prediction == "machine":
            machine.append(file.filename)
            
        '''
        print(
            "--------------------\nThis image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_prediction, 100 * np.max(score))
        )
        '''
    return handwritten, machine, unknown

'''
Delete the static/user folder if exists (called when the website is loaded)
'''
def remove_user_folder():
    if os.path.isdir(UPLOAD_FOLDER):
        shutil.rmtree(UPLOAD_FOLDER)

'''
Create a new static/user folder
'''
def new_static_folder():
    if not os.path.isdir(UPLOAD_FOLDER):
        os.mkdir(UPLOAD_FOLDER)

'''
Validates whether the user file is the correct file type
'''
def is_accepted_extension(filename):
    if '.' in filename:
        extension = filename.rsplit('.', 1)[1].lower()
        if extension in ALLOWED_FILE_TYPE:
            return True
    return False


'''
Renders the homepage (where the user submits their images)
'''
@app.route('/', methods=['GET', 'POST'])
def upload_form():
    # Create a new static/user folder to hold new user's images
    remove_user_folder()
    new_static_folder()
    return render_template('input.html')

'''
Renders diisplay page once the sort button is pressed
'''
@app.route('/display', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # displays a message if no files were submitted
        if 'files[]' not in request.files:
            flash('No file part')
            return redirect(request.url)

        # retrieve files from website
        files = request.files.getlist('files[]')
        handwritten = []
        machine = []
        unknown = []

        # saves files to server (IT IS SAVED IN THE STATIC FOLDER)
        for file in files:
            if file and is_accepted_extension(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(UPLOAD_FOLDER, filename))

        handwritten, machine, unknown= produce_predictions(files)

        # opens output HTML and passes the lists of filenames      
        return render_template("display.html", handwritten = handwritten, machine = machine, unknown = unknown)

#app.run()
if __name__ == "__main__":
    app.run(host='127.0.0.1',port=5000,debug=True,threaded=True)