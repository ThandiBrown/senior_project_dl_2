import os
import shutil
import tensorflow as tf
import numpy as np
from flask import request
from werkzeug.utils import secure_filename

#UPLOAD_FOLDER = "/content/gdrive/MyDrive/spweb/static/user/"
UPLOAD_FOLDER = "static/user/"
ALLOWED_FILE_TYPE = set(['png', 'jpg', 'jpeg'])
class_names = ['handwritten', 'machine']


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
returns three lists containing the DL model prediction results
'''
def produce_predictions(files_list, new_model):
    ''' Placeholder '''

    handwritten = []
    machine = []
    unknown = []
            
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
Validates whether the user file is the correct file type
'''
def is_accepted_extension(filename):
    if '.' in filename:
        extension = filename.rsplit('.', 1)[1].lower()
        if extension in ALLOWED_FILE_TYPE:
            return True
    return False

'''
Performs file handling for the user's input and and runs produce_predictions
Returns the results of produce_predictions
'''
def produce_results(request, new_model):
    # retrieve files from website
    files = request.files.getlist('files[]')
    handwritten = []
    machine = []
    unknown = []
    
    # saves files to server (IT IS SAVED IN THE STATIC FOLDER)
    skip_files = 0
    ignore_files = []

    for file in files:
        if file and is_accepted_extension(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
        else:
            skip_files += 1
            ignore_files.append(file)
    
    # If there are any valid file types to evaluate
    if len(files) - skip_files > 0:
        if skip_files:
            files = [i for i in files if i not in ignore_files]
        handwritten, machine, unknown = produce_predictions(files, new_model)
    
    return handwritten, machine, unknown