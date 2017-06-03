import tensorflow as tf
import cv2
import numpy as np
from timeit import default_timer as timer

import sys
sys.path.append("../")

from preprocessing import ssd_vgg_preprocessing
from nets import ssd_vgg_512, np_methods
from notebooks import visualization

slim = tf.contrib.slim

class_names = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "monitor"]

# A file path to a video to be tested on. Can also be a number, 
# in which case the webcam with the same number (i.e. 0) is used instead
video_path = 1

select_threshold=0.5
nms_threshold=.45

# TensorFlow session: grow memory when needed.
#gpu_options = tf.GPUOptions(allow_growth=True)
#config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
#isess = tf.InteractiveSession(config=config)
#isess = tf.InteractiveSession()
isess = tf.Session()

# Input placeholder.
net_shape = (512, 512)
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))

# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)

# Define the SSD model.
ssd_net = ssd_vgg_512.SSDNet()
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False)

# Restore SSD model.
ckpt_filename = '../checkpoints/VGG_VOC0712_SSD_512x512_ft_iter_120000.ckpt'
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)

# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)


########################################################

# Create unique and somewhat visually distinguishable bright
# colors for the different classes.
num_classes = len(class_names)
class_colors = []
for i in range(0, num_classes):
    # This can probably be written in a more elegant manner
    hue = 255*i/num_classes
    col = np.zeros((1,1,3)).astype("uint8")
    col[0][0][0] = hue
    col[0][0][1] = 128 # Saturation
    col[0][0][2] = 255 # Value
    cvcol = cv2.cvtColor(col, cv2.COLOR_HSV2BGR)
    col = (int(cvcol[0][0][0]), int(cvcol[0][0][1]), int(cvcol[0][0][2]))
    class_colors.append(col) 


vid = cv2.VideoCapture(video_path)
#vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not vid.isOpened():
    raise IOError(("Couldn't open video file or webcam. If you're "
    "trying to open a webcam, make sure you video_path is an integer!"))

# Compute aspect ratio of video     
#vidw = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
#vidh = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
#vidar = vidw/vidh

# Skip frames until reaching start_frame
#if start_frame > 0:
#    vid.set(cv2.CAP_PROP_POS_MSEC, start_frame)
    
accum_time = 0
curr_fps = 0
fps = "FPS: ??"
prev_time = timer()

while (vid.isOpened()):
    retval, orig_image = vid.read()
    if not retval:
        print("Done!")
        break
    
    to_draw = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)

    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img = isess.run(
        [image_4d, predictions, localisations, bbox_img],
        feed_dict={img_input: to_draw})

    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
            rpredictions, rlocalisations, ssd_anchors,
            select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)

    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
            rpredictions, rlocalisations, ssd_anchors,
            select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)

    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)

    height = to_draw.shape[0]
    width = to_draw.shape[1]

    for i in range(rclasses.shape[0]):
        cls_id = int(rclasses[i])
        if cls_id >= 0:
            score = rscores[i]
            ymin = int(rbboxes[i, 0] * height)
            xmin = int(rbboxes[i, 1] * width)
            ymax = int(rbboxes[i, 2] * height)
            xmax = int(rbboxes[i, 3] * width)
            cv2.rectangle(to_draw, (xmin, ymin), (xmax, ymax), class_colors[cls_id], 2)
            
            #text_top = (xmin, ymin-10)
            #text_bot = (xmin + 80, ymin + 5)
            #text_pos = (xmin + 5, ymin)

            text_top = (xmin, ymin)
            text_bot = (xmin + 80, ymin + 15)
            text_pos = (xmin + 5, ymin + 10)
            cv2.rectangle(to_draw, text_top, text_bot, class_colors[cls_id], -1)
            
            text = class_names[cls_id] + " " + ('%.2f' % score)
            cv2.putText(to_draw, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
    
    # Calculate FPS
    # This computes FPS for everything, not just the model's execution 
    # which may or may not be what you want
    curr_time = timer()
    exec_time = curr_time - prev_time
    prev_time = curr_time
    accum_time = accum_time + exec_time
    curr_fps = curr_fps + 1
    if accum_time > 1:
        accum_time = accum_time - 1
        fps = "FPS: " + str(curr_fps)
        curr_fps = 0
    
    # Draw FPS in top left corner
    cv2.rectangle(to_draw, (0,0), (50, 17), (255,255,255), -1)
    cv2.putText(to_draw, fps, (3,10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)

    wintext = "SSD result " + str(width) + "x" + str(height)
    cv2.imshow(wintext, to_draw)

    k = cv2.waitKey(1)
    if k==27:    # Esc key to stop
        break

# Release everything if job is finished
vid.release()
cv2.destroyAllWindows()





