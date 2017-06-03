'''
Change the hardcoded host urls below with your own hosts. 
Run like this: 

pc-01$ python my_test3.py --job_name="ps" --task_index=0
pc-02$ python my_test3.py --job_name="worker" --task_index=0
pc-03$ python my_test3.py --job_name="worker" --task_index=1
pc-04$ python my_test3.py --job_name="worker" --task_index=2

'''

import tensorflow as tf
import time
import math
from tensorflow.examples.tutorials.mnist import input_data

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
        tf.set_random_seed(1)
        # Variables of the hidden layer
        hid_w = tf.Variable(tf.truncated_normal([784, 100], stddev=1.0 / 28), name="hid_w")
        hid_b = tf.Variable(tf.zeros([100]), name="hid_b")

        # Variables of the softmax layer
        sm_w = tf.Variable(tf.truncated_normal([100, 10], stddev=1.0 / math.sqrt(100)), name="sm_w")
        sm_b = tf.Variable(tf.zeros([10]), name="sm_b")

        x = tf.placeholder(tf.float32, [None, 784])
        y_ = tf.placeholder(tf.float32, [None, 10])

        hid_lin = tf.nn.xw_plus_b(x, hid_w, hid_b)
        hid = tf.nn.relu(hid_lin)

        y = tf.nn.softmax(tf.nn.xw_plus_b(hid, sm_w, sm_b))
        loss = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))

        global_step = tf.Variable(0)

        train_op = tf.train.AdagradOptimizer(0.01).minimize(loss, global_step=global_step)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        saver = tf.train.Saver()
        summary_op = tf.summary.merge_all()
        init_op = tf.global_variables_initializer()

    # Create a "supervisor", which oversees the training process.
    sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                logdir=None,
                                init_op=init_op,
                                summary_op=summary_op,
                                saver=saver,
                                global_step=global_step,
                                save_model_secs=600)

    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

    # The supervisor takes care of session initialization, restoring from
    # a checkpoint, and closing when done or an error occurs.
    with sv.managed_session(server.target) as sess:
        # Loop until the supervisor shuts down or 1000000 steps have completed.
        step = 0
        while not sv.should_stop() and step < 50000:
            # Run a training step asynchronously.
            # See `tf.train.SyncReplicasOptimizer` for additional details on how to
            # perform *synchronous* training.

            batch_xs, batch_ys = mnist.train.next_batch(100)
            train_feed = {x: batch_xs, y_: batch_ys}

            _, step = sess.run([train_op, global_step], feed_dict=train_feed)

            if step % 100 == 0:
                print("Done step %d" % step)
                print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

    # Ask for all the services to stop.
    sv.stop()
    print("--- done ---")

