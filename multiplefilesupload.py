import os
from flask_ngrok import run_with_ngrok
from flask import Flask, request, render_template

import tensorflow as tf

import helper

app = Flask(__name__)

new_model = tf.keras.models.load_model('C:/Users/thanb/Documents/CodeFiles/model/hm_model7')

'''
Renders the homepage (where the user submits their images)
'''
@app.route('/', methods=['GET', 'POST'])
def upload_form():
    # Create a new static/user folder to hold new user's images
    helper.remove_user_folder()
    helper.new_static_folder()
    return render_template('input.html')

'''
Renders display page once the sort button is pressed
'''
@app.route('/display', methods=['POST'])
def display_results():
    if request.method == 'POST':
        files = request.files.getlist('files[]')

        if len(files) > 500:
            return render_template('too_many.html')

        handwritten, machine, unknown = helper.produce_results(files, new_model)
        # opens output HTML and passes the lists of filenames      
        return render_template("display.html", handwritten = handwritten, machine = machine, unknown = unknown)

#app.run()
if __name__ == "__main__":
    app.run(host='127.0.0.1',port=5000,debug=True,threaded=True)