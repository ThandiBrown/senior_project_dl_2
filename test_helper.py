import unittest
import helper
import os
import tensorflow as tf
from werkzeug.datastructures import FileStorage

class TestHelper(unittest.TestCase):

    def test_remove_user_folder(self):
        helper.UPLOAD_FOLDER = "test_folder/"
        if not os.path.isdir(helper.UPLOAD_FOLDER):
            os.mkdir(helper.UPLOAD_FOLDER)
        helper.remove_user_folder()
        self.assertTrue(not os.path.isdir(helper.UPLOAD_FOLDER))
        

    def test_new_static_folder(self):
        helper.UPLOAD_FOLDER = "test_folder/"
        self.assertTrue(not os.path.isdir(helper.UPLOAD_FOLDER))
        helper.new_static_folder()
        self.assertTrue(os.path.isdir(helper.UPLOAD_FOLDER))

    def test_is_accepted_extension(self):
        filename = "lovely.png"
        self.assertTrue(helper.is_accepted_extension(filename))        
        filename = "lovely.txt"
        self.assertFalse(helper.is_accepted_extension(filename))

    def test_produce_predictions(self):
        helper.UPLOAD_FOLDER = ""
        new_model = tf.keras.models.load_model('C:/Users/thanb/Documents/CodeFiles/model/hm_model7')
        
        files_list = []

        h, m, u = helper.produce_predictions(files_list, new_model)
        has_empty_list_value = len(h) == 0 or len(m) == 0 or len(u) == 0
        self.assertTrue(has_empty_list_value)
        
        
        with open('static/hwd.jpg', 'rb') as fp:
            file = FileStorage(fp)
        files_list.append(file)

        h, m, u = helper.produce_predictions(files_list, new_model)
        has_list_value = len(h)>0 or len(m)>0 or len(u)>0
        self.assertTrue(has_list_value)


if __name__ == '__main__':
    unittest.main()