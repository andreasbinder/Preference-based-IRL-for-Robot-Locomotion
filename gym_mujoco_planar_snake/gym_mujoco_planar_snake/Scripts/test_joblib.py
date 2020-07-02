import tensorflow as tf
import joblib

var_list = [tf.Variable(i, name=str(i)) for i in range(10)]

#tf.InteractiveSession()

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    #print(var_list[0].eval())

    model = { var.name : var.eval() for var in var_list}

    print(model)

    joblib.dump(model, "/home/andreas/Desktop/Checkpoints/file")

    restored = joblib.load('/home/andreas/Desktop/Checkpoints/file')

    print(type(restored))