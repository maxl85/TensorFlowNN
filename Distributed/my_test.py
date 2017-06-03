'''
Change the hardcoded host urls below with your own hosts. 
Run like this: 

pc-01$ python example.py --job_name="ps" --task_index=0 
pc-02$ python example.py --job_name="worker" --task_index=0 
pc-03$ python example.py --job_name="worker" --task_index=1 
pc-04$ python example.py --job_name="worker" --task_index=2 

'''

import tensorflow as tf
import sys
import time
from tensorflow.examples.tutorials.mnist import input_data

# Create a cluster from the parameter server and worker hosts.
parameter_servers = ["localhost:2220"]
workers = ["localhost:2221",
           "localhost:2222",
           "localhost:2223"]
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
    print("--- Parameter Server Ready ---")
elif FLAGS.job_name == "worker":
    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(
                   worker_device="/job:worker/task:%d" % FLAGS.task_index,
                   cluster=cluster)):

        # Build model...
        loss = ...

        global_step = tf.contrib.framework.get_or_create_global_step()

        train_op = tf.train.AdagradOptimizer(0.01).minimize(loss, global_step=global_step)

    # The StopAtStepHook handles stopping after running given steps.
    hooks=[tf.train.StopAtStepHook(last_step=1000000)]

    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=(FLAGS.task_index == 0),
                                           checkpoint_dir="./tmp/train_logs",
                                           hooks=hooks) as mon_sess:

        while not mon_sess.should_stop():
            # Run a training step asynchronously.
            # See `tf.train.SyncReplicasOptimizer` for additional details on how to
            # perform *synchronous* training.
            # mon_sess.run handles AbortedError in case of preempted PS.
            mon_sess.run(train_op)

    print("--- done ---")
