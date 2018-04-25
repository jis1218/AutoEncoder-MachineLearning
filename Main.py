# coding: utf-8
'''
Created on 2018. 4. 24.

@author: Insup Jung
'''

import tensorflow as tf
import argparse
from tensorflow.examples.tutorials.mnist import input_data
from AutoEncoder.AE import *

if __name__ == '__main__':
    
    # 명령줄 옵션을 파싱하는 객체
#     parser = argparse.ArgumentParser(description='Test various optimization strategies')
#     parser.add_argument('n_code', nargs=1, type=str) #nargs : 스위치나 파라미터가 받을 수 있는 값의 개수, type : 파싱하여 저장할 때 타입을 변경할 수 있음
#     args = parser.parse_args()
#     n_code = args.n_code[0]
    learning_rate = 0.0001
    training_epochs = 20
    display_step = 1
    batch_size = 100
    
    mnist = input_data.read_data_sets("data/", one_hot=True)
    
    with tf.Graph().as_default():
        with tf.variable_scope("autoencoder_model"):
            
            x = tf.placeholder("float", [None, 784])
            phase_train = tf.placeholder(tf.bool)
            
            code = encoder(x, 2, phase_train)
            
            output = decoder(code, 2, phase_train)
            
            cost, train_summary_op = loss(output, x)
            
            global_step = tf.Variable(0, name='global_step', trainable=False)
            
            train_op = training(cost, global_step)
            
            eval_op, in_im_op, out_im_op, val_summary_op = evaluate(output, x)
            
            summary_op = tf.summary.merge_all()
            
            saver = tf.train.Saver(max_to_keep=200)
            
            sess = tf.Session()
            
            train_writer = tf.summary.FileWriter("mnist_autoencoder_hidden=" + str(2) + "_logs/", graph=sess.graph)
            
            val_writer = tf.summary.FileWriter("mnist_autoencoder_hidden=" + str(2) + "_logs/", graph=sess.graph)
            
            init_op = tf.global_variables_initializer()
            
            sess.run(init_op)
            
            for epoch in range(training_epochs):
                avg_cost = 0
                total_batch = int(mnist.train._num_examples/batch_size) #example 개수 55000개 / batch_size 100 = 550이 나옴
                
                for i in range(total_batch):
                    mbatch_x, mbatch_y = mnist.train.next_batch(batch_size)
                    
                    _, new_cost, train_summary = sess.run([train_op, cost, train_summary_op], feed_dict={x:mbatch_x, phase_train: True})
                    train_writer.add_summary(train_summary, sess.run(global_step))
                    
                    avg_cost += new_cost/total_batch
                    
                if epoch % display_step == 0:
                    print ("Epoch:", '%04d' % (epoch+1), "cost =", "{:.9f}".format(avg_cost))
                    
                    train_writer.add_summary(train_summary, sess.run(global_step))
                    val_images = mnist.validation.images
                    validation_loss, in_im, out_im, val_summary = \
                    sess.run([eval_op, in_im_op, out_im_op, val_summary_op], feed_dict={x: val_images, phase_train: False})
                    
                    val_writer.add_summary(in_im, sess.run(global_step))
                    val_writer.add_summary(out_im, sess.run(global_step))
                    val_writer.add_summary(val_summary, sess.run(global_step))
                    
                    print ("Validation Loss:", validation_loss)
                    
                    saver.save(sess, "mnist_autoencoder_hidden=" + str(2) + "_logs/model-checkpoint-" + '%04d' % (epoch+1), global_step=global_step)
                    
                    print ("Optimization Finished")
                    
                    test_loss = sess.run(eval_op, feed_dict={x:mnist.test.images, phase_train:False})
                    #print ("test Loss:", loss)
    
    pass