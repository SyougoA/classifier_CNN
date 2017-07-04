import cv2
import os
import numpy as np
import tensorflow as tf


NUM_CLASSES = 4
IMG_SIZE = 28
COLOR_CHANNELS = 3
IMG_PIXELS = IMG_SIZE * IMG_SIZE * COLOR_CHANNELS


def inference(images_placeholder, keep_prob):
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(x, w):
        return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    x_image = tf.reshape(images_placeholder, [-1, IMG_SIZE, IMG_SIZE, COLOR_CHANNELS])

    with tf.name_scope('conv1') as scope:
        w_conv1 = weight_variable([5, 5, COLOR_CHANNELS, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)

    with tf.name_scope('pool1') as scope:
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope('conv2') as scope:
        w_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)

    with tf.name_scope('pool2') as scope:
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope('fc1') as scope:
        w_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.name_scope('fc2') as scope:
        w_fc2 = weight_variable([1024, NUM_CLASSES])
        b_fc2 = bias_variable([NUM_CLASSES])

    with tf.name_scope('softmax') as scope:
        y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

    return y_conv


images_placeholder = tf.placeholder("float", shape=(None, IMG_PIXELS))
keep_prob = tf.placeholder("float")
logits = inference(images_placeholder, keep_prob)
sess = tf.InteractiveSession()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
saver.restore(sess, "model.ckpt")


def classify_character(file_name):
    path = os.getcwd()
    image = cv2.imread(path + "/" + file_name)
    cascade = cv2.CascadeClassifier("使用するxml")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    facerect = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))

    if len(facerect) > 0:
        for i in range(len(facerect)):
            rect = facerect[i]
            oriimg = image[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
            img = cv2.resize(oriimg, (28, 28))
            test_image = np.asarray([img.flatten().astype(np.float32) / 255.0])
            pred = np.argmax(logits.eval(feed_dict={images_placeholder: [test_image[0]],keep_prob: 1.0})[0])
            if pred == 0:
                col = (0, 0, 255)
            elif pred == 1:
                col = (0, 255, 255)
            elif pred == 2:
                col = (128, 0, 0)
            else:
                col = (128, 128, 255)
            cv2.rectangle(image, tuple(rect[0:2]), tuple(rect[0:2] + rect[2:4]), col, thickness=2)

    cv2.imwrite("classified/" + file_name, image)
    return "%dキャラクター検知！" % len(facerect)


if __name__ == "__main__":
    file_name = input("検知画像を入力してください :")
    classify_character(file_name)
    img = cv2.imread("classified/" + file_name)
    cv2.imshow("img", img)
    cv2.waitKey()