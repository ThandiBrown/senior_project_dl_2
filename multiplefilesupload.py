#----------IGNORE----------
import shutil
import os
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename


import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plte

app=Flask(__name__)

app.secret_key = "secret key"
#app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

UPLOAD_FOLDER = "static/user/"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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

        # if confidence percent is this low make it unknown
        if class_prediction == "handwritten":
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

def remove_static_folder():
    if os.path.isdir(UPLOAD_FOLDER):
        shutil.rmtree(UPLOAD_FOLDER)

def new_static_folder():
    if not os.path.isdir(UPLOAD_FOLDER):
        os.mkdir(UPLOAD_FOLDER)

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
#------------IGNORE OVER---------

# This is the first website called when you go to the 5000 website
@app.route('/')
def upload_form():
    remove_static_folder()
    new_static_folder()
    return render_template('input.html')

# This is the website called once you press the submit button (displays images)
@app.route('/', methods=['POST'])# SUBMIT BUTTON
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
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        handwritten, machine, unknown= produce_predictions(files)

        # opens output HTML and passes the lists of filenames      
        return render_template("display.html", handwritten = handwritten, machine = machine, unknown = unknown)


if __name__ == "__main__":
    app.run(host='127.0.0.1',port=5000,debug=True,threaded=True)