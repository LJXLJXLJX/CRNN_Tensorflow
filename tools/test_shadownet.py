#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-9-29 下午3:56
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/CRNN_Tensorflow
# @File    : test_shadownet.py
# @IDE: PyCharm Community Edition
"""
Use shadow net to recognize the scene text of a single image
"""
import argparse
import os.path as ops
import os

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.tools.freeze_graph import freeze_graph
import matplotlib.pyplot as plt
import glog as logger

from config import global_config
from crnn_model import crnn_model
from data_provider import tf_io_pipline_tools
from local_utils.custom_ctc_decoder import ctc_decode

from time import time

CFG = global_config.cfg


def init_args():
    """
    :return: parsed arguments and (updated) config.cfg object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str,
                        help='Path to the image to be tested',
                        default='./data/test_images/test_01.jpg')
    parser.add_argument('--weights_path', type=str,
                        help='Path to the pre-trained weights to use',
                        default='./model/crnn_syn90k/shadownet.ckpt')
    parser.add_argument('-c', '--char_dict_path', type=str,
                        help='Directory where character dictionaries for the dataset were stored',
                        default='./data/char_dict/char_dict.json')
    parser.add_argument('-o', '--ord_map_dict_path', type=str,
                        help='Directory where ord map dictionaries for the dataset were stored',
                        default='./data/char_dict/ord_map.json')
    parser.add_argument('-v', '--visualize', type=args_str2bool, nargs='?', const=True,
                        help='Whether to display images',
                        default=True)

    return parser.parse_args()


def args_str2bool(arg_value):
    """

    :param arg_value:
    :return:
    """
    if arg_value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True

    elif arg_value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def recognize(image_path, weights_path, char_dict_path, ord_map_dict_path, is_vis):
    # def recognize(weights_path, char_dict_path, ord_map_dict_path, is_vis):
    """
    :param image_path:
    :param weights_path:
    :param char_dict_path:
    :param ord_map_dict_path:
    :param is_vis:
    :return:
    """

    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), 1)
    new_heigth = 32
    rate = new_heigth / image.shape[0]
    new_width = int(rate * image.shape[1])
    image = cv2.resize(image, (new_width, new_heigth), interpolation=cv2.INTER_LINEAR)
    image_vis = image
    image = np.array(image, np.float32) / 127.5 - 1.0
    inputdata = tf.placeholder(
        dtype=tf.float32,
        shape=[None, new_heigth, None, CFG.ARCH.INPUT_CHANNELS],
        name='input'
    )

    net = crnn_model.ShadowNet(
        phase='test',
        hidden_nums=CFG.ARCH.HIDDEN_UNITS,
        layers_nums=CFG.ARCH.HIDDEN_LAYERS,
        num_classes=CFG.ARCH.NUM_CLASSES
    )

    inference_ret = net.inference(
        inputdata=inputdata,
        name='shadow_net',
        reuse=False
    )
    # config tf saver
    saver = tf.train.Saver()

    decodes, _ = tf.nn.ctc_beam_search_decoder(
        inputs=inference_ret,
        sequence_length=new_width // 4 * np.ones(1),
        merge_repeated=False
    )

    codec = tf_io_pipline_tools.TextFeatureIO(
        char_dict_path=char_dict_path,
        ord_map_dict_path=ord_map_dict_path
    ).reader

    # config tf session
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TEST.TF_ALLOW_GROWTH

    sess = tf.Session(config=sess_config)

    with sess.as_default():
        saver.restore(sess=sess, save_path=weights_path)

        # debug_1 = sess.run(inference_ret, feed_dict={inputdata: [image]})
        # debug_1 = debug_1.argmax(axis=2)[:,0]
        # print(debug_1)

        preds = sess.run(decodes, feed_dict={inputdata: [image]})

        preds = codec.sparse_tensor_to_str(preds[0])

        logger.info('Predict image {:s} result {:s}'.format(
            ops.split(image_path)[1], preds[0])
        )

        if is_vis:
            plt.figure('CRNN Model Demo')
            plt.imshow(image_vis[:, :, (2, 1, 0)])
            plt.show()
    sess.close()
    return


def define_graph():
    inputdata = tf.placeholder(
        dtype=tf.float32,
        shape=[1, 32, None, CFG.ARCH.INPUT_CHANNELS],
        name='input'
    )

    net = crnn_model.ShadowNet(
        phase='test',
        hidden_nums=CFG.ARCH.HIDDEN_UNITS,
        layers_nums=CFG.ARCH.HIDDEN_LAYERS,
        num_classes=CFG.ARCH.NUM_CLASSES
    )

    inference_ret = net.inference(
        inputdata=inputdata,
        name='shadow_net',
        reuse=False
    )
    with tf.Session() as sess:
        graph_def = sess.graph.as_graph_def()
        with tf.gfile.FastGFile('../model/chinese/test.pb', 'wb') as f:
            f.write(graph_def.SerializeToString())


def print_pb_debug(pb_path):
    with tf.Session() as sess:
        with open(pb_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            print(graph_def)


def freezeGraph():
    freeze_graph(input_graph='../model/chinese/test.pb',  # =some_graph_def.pb
                 input_saver="",
                 input_checkpoint='../model/chinese/shadownet_2019-03-28-11-58-32.ckpt-200000',
                 checkpoint_version=2,
                 output_graph='../model/chinese/out.pb',
                 input_binary=True,
                 restore_op_name="save/restore_all",
                 filename_tensor_name="save/Const:0",
                 initializer_nodes="",
                 variable_names_whitelist="",
                 variable_names_blacklist="",
                 input_meta_graph="",
                 saved_model_tags='serve',
                 clear_devices=True,
                 output_node_names='shadow_net/sequence_rnn_module/transpose_time_major',
                 )


def pb_recognize(pb_path, char_dict_path, ord_map_dict_path):
    codec = tf_io_pipline_tools.TextFeatureIO(
        char_dict_path=char_dict_path,
        ord_map_dict_path=ord_map_dict_path
    ).reader

    with tf.gfile.FastGFile(pb_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

    with tf.Session() as session:
        new_heigth = 32
        while True:
            path = input('please input the image path\n')
            if path == '':
                break
            if not ops.exists(path):
                print('invalid path')
                continue
            img_path_list = []
            if ops.isfile(path):
                img_path_list.append(path)
            elif ops.isdir(path):
                files = os.listdir(path)
                for file in files:
                    img_path_list.append(ops.join(path, file))

            for img_path in img_path_list:
                image = cv2.imdecode(np.fromfile(img_path, np.uint8), 1)
                rate = new_heigth / image.shape[0]
                new_width = int(rate * image.shape[1])
                image = cv2.resize(image, (new_width, new_heigth), interpolation=cv2.INTER_LINEAR)
                image_vis = image
                image = np.array(image, np.float32) / 127.5 - 1.0
                image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
                prediction_tensor = session.graph.get_tensor_by_name(
                    'shadow_net/sequence_rnn_module/transpose_time_major:0')

                # ----------------tf-----------------------------------

                # decodes, _ = tf.nn.ctc_beam_search_decoder(
                #     inputs=prediction_tensor,
                #     sequence_length=new_width // 4 * np.ones(1),
                #     merge_repeated=False
                # )
                # out = session.run(decodes, {'input:0': image})
                # text = codec.sparse_tensor_to_str(out[0])
                # print(text[0])

                # -----------------------my implement----------------------------------------
                out = session.run(prediction_tensor, {'input:0': image})
                out = out.argmax(axis=2)[:, 0]
                out = ctc_decode(out)
                text = codec.array_to_str(out)

                print(text)


if __name__ == '__main__':
    pass
    # # init images
    # args = init_args()
    #
    # # detect images
    # recognize(
    #     image_path=args.image_path,
    #     weights_path=args.weights_path,
    #     char_dict_path=args.char_dict_path,
    #     ord_map_dict_path=args.ord_map_dict_path,
    #     is_vis=args.visualize
    # )
    #############################
    #
    # recognize(
    #     image_path='F:\Project\ocr-demo\测试图片-识别/TIM截图20190309134248.jpg',
    #     weights_path='../model/chinese/shadownet_2019-03-28-11-58-32.ckpt-200000',
    #     char_dict_path='../data/char_dict/char_dict.json',
    #     ord_map_dict_path='../data/char_dict/ord_map.json',
    #     is_vis=True
    # )

    # define_graph()
    # freezeGraph()

    pb_recognize(
        '../model/chinese/out.pb',
        '../data/char_dict/char_dict.json',
        '../data/char_dict/ord_map.json')
