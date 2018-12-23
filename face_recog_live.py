import logging
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework import ops
import cv2

from scipy import misc
import detect_face

logger = logging.getLogger(__name__)


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


class ImageClass():
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)

def main():
    model_path = "models/20170511-185253.pb"
    # classifier_output_path = "/mnt/softwares/acv_project_code/Code/classifier_rf1_team.pkl"
    classifier_output_path = "models/classifier_rf4.pkl"
    #classifier_output_path = "/mnt/softwares/acv_project_code/Code/classfier_path/classifier_svm.pkl"

    with gfile.FastGFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embedding_layer = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    gpu_memory_fraction = 1
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess1 = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        # sess1 = tf.Session(config=tf.ConfigProto(device_count = {'GPU': 0}))
        with sess1.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess1, None)

    model, class_names = pickle.load(open(classifier_output_path, 'rb'), encoding='latin1')
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture('/home/lokender/Downloads/orig_faces/videos/nayeem.mp4')
    # cap = cv2.VideoCapture('/home/lokender/Downloads/orig_faces/videos/lokender.mp4')
    fno = 0
    det_name = []
    det_prob =[]
    bbs = []
    while (~(cv2.waitKey(1) & 0xFF == ord('q'))):

        # image2 = cv2.imread("/home/lokender/Downloads/T1/both/IMG_20171115_150720.jpg")
        # image2.set_shape((480, 640, 3))
        # image2= cv2.resize(image2, (640,480))
        ret, image2 = cap.read()
        image2 = cv2.resize(image2, (320, 240))
        if fno % 5 == 0:

            # image2 = cv2.imread("/home/lokender/Downloads/T1/both/IMG_20171115_150720.jpg")
            # image2.set_shape((480, 640, 3))
            # image2= cv2.resize(image2, (640,480))

            # image2 = cv2.imread("/home/lokender/Downloads/T1/both/IMG_20171115_150720.jpg")
            # image2.set_shape((480, 640, 3))

            # image2= cv2.resize(image2, (640,480))
            print(fno)
            # image2=rotate_bound(image1,90)
            # image2 = cv2.imread('/home/lokender/Downloads/acv_tmp/tm_al/tmp/frame_0.png', cv2.IMREAD_COLOR)

            # cv2.imwrite("/home/lokender/Downloads/acv_tmp/tm/tmp/frame.png", image2)
            image_size = 160
            margin = 32
            detect_multiple_faces = True


            minsize = 20  # minimum size of face
            threshold = [0.6, 0.7, 0.7]  # three steps's threshold
            factor = 0.709  # scale factor

            img = image2[:, :, 0:3]

            bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
            nrof_faces = bounding_boxes.shape[0]
            print(nrof_faces)
            if nrof_faces > 0:
                det = bounding_boxes[:, 0:4]
                det_arr = []
                img_size = np.asarray(img.shape)[0:2]
                if nrof_faces > 1:
                    if detect_multiple_faces:
                        for i in range(nrof_faces):
                            det_arr.append(np.squeeze(det[i]))
                    else:
                        bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                        img_center = img_size / 2
                        offsets = np.vstack(
                            [(det[:, 0] + det[:, 2]) / 2 - img_center[1], (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                        offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                        index = np.argmax(
                            bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
                        det_arr.append(det[index, :])
                else:
                    det_arr.append(np.squeeze(det))
                det_name = []
                det_prob =[]
                bbs = []
                for i, det in enumerate(det_arr):
                    det = np.squeeze(det)
                    bb = np.zeros(4, dtype=np.int32)
                    bb[0] = np.maximum(det[0] - margin / 2, 0)
                    bb[1] = np.maximum(det[1] - margin / 2, 0)
                    bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
                    bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
                    cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                    bbs.append(bb)
                    scaled = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
                    # nrof_successfully_aligned += 1
                    # output_filename_n = "{}_{}.{}".format(output_filename.split('.')[0], i,
                    #                                      output_filename.split('.')[-1])
                    # misc.imsave(output_filename_n, scaled)
                    # config=tf.ConfigProto(device_count = {'GPU': 0})
                    with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=1)))) as sess:
                        image_paths = ['/home/nayeem/Desktop/acv_live_face_recognition_project/src/images/frame_0.png']
                        image_size = 160
                        batch_size = 1
                        num_threads = 1
                        num_epochs = 1

                        label_list = [0]
                        images = ops.convert_to_tensor(image_paths, dtype=tf.string)
                        labels = ops.convert_to_tensor(label_list, dtype=tf.int32)

                        # Makes an input queue
                        input_queue = tf.train.slice_input_producer((images, labels),
                                                                    num_epochs=num_epochs, shuffle=False, )

                        images_labels = []
                        image = tf.convert_to_tensor(scaled)
                        label = input_queue[1]
                        # image = tf.random_crop(image, size=[image_size, image_size, 3])
                        # image.set_shape((image_size, image_size, 3))
                        image = tf.image.per_image_standardization(image)
                        images_labels.append([image, label])
                        num_threads = 16
                        images, labels = tf.train.batch_join(images_labels,
                                                             batch_size=batch_size,
                                                             capacity=4 * num_threads,
                                                             enqueue_many=False,
                                                             allow_smaller_final_batch=True)

                        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
                        sess.run(init_op)

                        coord = tf.train.Coordinator()
                        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

                        emb_array = None
                        batch_images, batch_labels = sess.run([images, labels])
                        emb = sess.run(embedding_layer,
                                       feed_dict={images_placeholder: batch_images, phase_train_placeholder: False})

                        emb_array = np.concatenate([emb_array, emb]) if emb_array is not None else emb

                        coord.request_stop()
                        coord.join(threads=threads)

                        predictions = model.predict_proba(emb_array, )
                        best_class_indices = np.argmax(predictions, axis=1)
                        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

                        for ji in range(len(best_class_indices)):
                            print(
                                '%4d  %s: %.3f' % (
                                ji, class_names[best_class_indices[ji]], best_class_probabilities[ji]))
                            det_name.append(class_names[best_class_indices[ji]])
                            det_prob.append(best_class_probabilities[ji])

        colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255], [255, 0, 255]]
        for jk in range(len(det_name)):
            # print jk
            bbt = bbs[jk]
            if det_prob[jk]>=0.5:
                cv2.rectangle(image2, (bbt[0], bbt[1]), (bbt[0] + (bbt[2] - bbt[0]), bbt[1] + (bbt[3] - bbt[1])),
                              colors[jk], 2)
                cv2.putText(image2, det_name[jk], (bbt[0] + (bbt[2] - bbt[0]) + 10, bbt[1] + (bbt[3] - bbt[1])), 0, 0.5,
                            colors[jk])
        cv2.imshow('fr', image2)
        fno = fno + 1
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()