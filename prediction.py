#!/usr/bin/env python
PKG = 'numpy_tutorial'
import roslib; roslib.load_manifest(PKG)
import rospy
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
import tensorflow as tf
import numpy as np
from std_msgs.msg import String
from std_msgs.msg import Int32
from numpy_tutorial.msg import Frequency
import datetime

class Predictor():
    def __init__(self):
        self.t1 = datetime.datetime.now()
        self.gpu_initializer()
        self.t2 = datetime.datetime.now()
        self.subscriber = rospy.Subscriber("send_image_data", numpy_msg(Floats),
                                           self.callback)
        self.pub = rospy.Publisher('prediction', Frequency, queue_size=10)
        #self.model = self.load_model("/home/mendez/PycharmProjects/pythonProject1/saved_model/model_1")
        #self.model_name = "model_1"
        self.model = self.load_model("/home/mendez/EfficientNetB7_model")
        self.model_name = "EfficientNetB7_model"
        self.class_names = ['chicken_curry', 'chicken_wings', 'fried_rice',
                            'grilled_salmon', 'hamburger', 'ice_cream',
                            'pizza', 'ramen', 'steak', 'sushi']

        self.frequency = Frequency()
        self.frequency.count = 0
        self.first_Loop = True
        self.load_model_time = datetime.datetime.now()
        self.average_prediction_time_exclude = 0
        self.average_prediction_time_include = 0
        self.list_pred_excl = []
        self.list_pred_incl = []
        self.list_numpy_pre = []

    
        

    def gpu_initializer(self):
        physical_devices = tf.config.list_physical_devices('GPU')
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], False)
        except:
            print("Invalid device")
            pass

    def load_model(self, model_path):
        return tf.keras.models.load_model(model_path)


    def preprocess_img(self, data):
        img = np.reshape(data, (224, 224, 3))
        tensor_img = tf.convert_to_tensor(img)
        return tensor_img


    def predict_class(self, model, img, class_names):
        pred = model.predict(tf.expand_dims(img, axis=0))


        if self.model_name == "EfficientNetB7_model":
            print("Index", pred.argmax())
            return 'pizza'


        if len(pred[0]) > 1: # check for multi-class
            pred_class = class_names[pred.argmax()] # if more than one output, take the max
        else:
            pred_class = class_names[int(tf.round(pred)[0][0])] # if only one output, round

        return pred_class


    def callback(self, data):
        if self.first_Loop:
            self.load_model_time = datetime.datetime.now()
            self.first_Loop = False
            print("LOADED TIME SUCCESSFUL STORED")

        t3 = datetime.datetime.now()
        tensor_img = self.preprocess_img(data.data)
        t4 = datetime.datetime.now()
        predicted_class = self.predict_class(self.model, tensor_img, self.class_names)
        
        self.frequency.count += 1
        self.frequency.className = predicted_class

        self.pub.publish(self.frequency)

        t5 = datetime.datetime.now()

        prediction_time_exc = t5 - t4
        prediction_time_incl = t5 - t3

        numpy_time_preprocess = t4 - t3

        self.list_pred_excl.append(prediction_time_exc.total_seconds())
        self.list_pred_incl.append(prediction_time_incl.total_seconds())
        self.list_numpy_pre.append(numpy_time_preprocess.total_seconds())

        
        #print("GPU INIT TIME T2-T1:", self.t2 - self.t1)
        #print("LOAD MODEL TIME excluding gpu init:", self.load_model_time - self.t2)
        print("LOAD MODEL TIME including gpu init:", self.load_model_time - self.t1)
        print("NUMPY PREPROCESS TIME:", numpy_time_preprocess)
        print("PREDICTION TIME Excluding numpy preprocess:", prediction_time_exc)
        print("PREDICTION TIME Including numpy preprocess:", prediction_time_incl)
        
        #print("ARRAY NUMPY preprocess:", self.list_numpy_pre)
        #print("ARRAY Excluding numpy preprocess:", self.list_pred_excl)
        #print("ARRAY Including numpy preprocess:", self.list_pred_incl)
        print("-------------------------------")


def main():
    Predictor()
    rospy.init_node('predictor')
    rospy.loginfo("Predictor Node is RUNNING")
    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
