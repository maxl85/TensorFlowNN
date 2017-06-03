'''
Change the hardcoded host urls below with your own hosts. 
Run like this: 

pc-01$ python my_test2.py --job_name="ps" --task_index=0
pc-02$ python my_test2.py --job_name="worker" --task_index=0
pc-03$ python my_test2.py --job_name="worker" --task_index=1
pc-04$ python my_test2.py --job_name="worker" --task_index=2

'''

import tensorflow as tf
#import sys
import time
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# Create a cluster from the parameter server and worker hosts.
parameter_servers = ["localhost:2220"]
workers = ["localhost:2221",
           "localhost:2222"]
#workers = ["localhost:2221"]
cluster = tf.train.ClusterSpec({"ps": parameter_servers, "worker": workers})

# input flags
tf.app.flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
FLAGS = tf.app.flags.FLAGS

# Create and start a server for the local task.
server = tf.train.Server(cluster,
                         job_name=FLAGS.job_name,
                         task_index=FLAGS.task_index)

if FLAGS.job_name == "ps":
    server.join()
elif FLAGS.job_name == "worker":
    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(
                   worker_device="/job:worker/task:%d" % FLAGS.task_index,
                   cluster=cluster)):

        # Build model...
        with tf.name_scope("input"):
            x = tf.placeholder(tf.float32, [None, 784], name="x-input")
            y_ = tf.placeholder(tf.float32, [None, 10], name="y-input")
        
        tf.set_random_seed(1)
        with tf.name_scope("weights"):
            W = tf.Variable(tf.zeros([784, 10]))
            b = tf.Variable(tf.zeros([10]))

        with tf.name_scope("model"):
            y = tf.matmul(x, W) + b

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
        global_step = tf.contrib.framework.get_or_create_global_step()
        train_op = tf.train.AdagradOptimizer(0.01).minimize(loss, global_step=global_step)

        with tf.name_scope("acc"):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # The StopAtStepHook handles stopping after running given steps.
    hooks=[tf.train.StopAtStepHook(last_step=2000)]

    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    begin_time = time.time()
    with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=(FLAGS.task_index == 0),
                                           checkpoint_dir=None,
                                           hooks=hooks) as sess:

        while not sess.should_stop():
            # Run a training step asynchronously.
            # See `tf.train.SyncReplicasOptimizer` for additional details on how to
            # perform *synchronous* training.
            # sess.run handles AbortedError in case of preempted PS.
            #sess.run(train_op)

            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(train_op, feed_dict={x: batch_xs, y_: batch_ys})
            
            print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
            

    print("--- done ---")
    print("Total Time: %3.2fs" % float(time.time() - begin_time))
    
