"""Preprocess the data of MSVD-QA."""
import os
import sys
import h5py
import pandas as pd
from pandas import Series, DataFrame
import tables
import tensorflow as tf
import json
import numpy as np
from util.preprocess import VideoVGGExtractor
from util.preprocess import VideoC3DExtractor
from util.preprocess import prune_embedding


def extract_vgg(video_directory, max_num_captions):
    """Extract VGG features."""
    vgg_features = list()
    # Session config.
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.gpu_options.visible_device_list = '0'

    resFile_video = '/home/yulia/github/Activity_net_whole/train10_qa.json'
    data = json.load(open(resFile_video))
    caption_info = data['caption_info']


    with tf.Graph().as_default(), tf.Session(config=sess_config) as sess:
        extractor = VideoVGGExtractor(max_num_captions, sess)
        video_id_train = open("video_id_train.txt","r")
        for video_id in video_id_train:
            video_id_cap = 'v_' + video_id[:-1]
            if video_id_cap in caption_info.keys():
                list_timestamps = caption_info[video_id_cap]['timestamps']
            video_path = os.path.join(video_directory, video_id[:-1] + '.avi')
            print('[VGG]', video_path)
            vgg_features.append(extractor.extract(video_path, list_timestamps, max_num_captions))
            # print(vgg_features[-1])
    video_id_train.close()
    return vgg_features

def extract_c3d(video_directory, max_num_captions, max_num_frames):
    """Extract C3D features."""
    c3d_features = list()
    # Session config.
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.gpu_options.visible_device_list = '0'
    resFile_video = '/home/yulia/github/Activity_net_whole/train10_qa.json'
    data = json.load(open(resFile_video))
    caption_info = data['caption_info']

    with tf.Graph().as_default(), tf.Session(config=sess_config) as sess:
        extractor = VideoC3DExtractor(max_num_captions, max_num_captions, max_num_frames, sess)
        video_id_train = open("video_id_train.txt","r")
        for video_id in video_id_train:
            video_id_cap = 'v_' + video_id[:-1]

            if video_id_cap in caption_info.keys():
                list_timestamps = caption_info[video_id_cap]['timestamps']
            # video_path = os.path.join(video_directory, video_id[2:-1] + '.mp4')
            video_path = os.path.join(video_directory, video_id[:-1] + '.avi')
            print('[C3D]', video_path)
            c3d_features.append(extractor.extract(video_path, list_timestamps))
            # print(c3d_features[-1])
    video_id_train.close()
    return c3d_features

def extract_video_feature(video_directory, max_num_captions, max_num_frames):
    """Extract video features(vgg, c3d) and store in hdf5 file."""
    feature_path_vgg = 'data/activity_net/vgg_feature.h5'
    feature_path_c3d = 'data/activity_net/c3d_feature.h5'

    h5file = tables.open_file(feature_path_vgg, 'w', 'Extracted video vgg of the ActivityNet dataset.')
    vgg_features = extract_vgg(video_directory, max_num_captions)
    h5file.create_array('/', 'vgg', vgg_features, 'vgg16 feature')
    h5file.close()

    h5file = tables.open_file(feature_path_c3d, 'w', 'Extracted video c3d of the ActivityNet dataset.')

    c3d_features = extract_c3d(video_directory, max_num_captions, max_num_frames)
    h5file.create_array('/', 'c3d', c3d_features, 'c3d feature')
    h5file.close()

def create_answerset(trainqa_path, answerset_path):
    """Generate 1000 answer set from train_qa.json.

    Args:
        trainqa_path: path to train_qa.json.
        answerset_path: generate answer set of mc_qa
    """
    train_qa = pd.read_json(trainqa_path)
    answer_freq = train_qa['answer'].value_counts()
    answer_freq = DataFrame(answer_freq.iloc[0:1000])
    answer_freq.to_csv(answerset_path, columns=[], header=False)


def create_vocab(trainqa_path, answerset_path, vocab_path):
    """Create the 4000 vocabulary based on questions in train split.
    3999 most frequent words and 1 <UNK>.

    Args:
        trainqa_path: path to train_qa.json.
        vocab_path: vocabulary file.
    """
    vocab = dict()
    train_qa = pd.read_json(trainqa_path)
    # remove question whose answer is not in answerset
    answerset = pd.read_csv(answerset_path, header=None)[0]
    train_qa = train_qa[train_qa['answer'].isin(answerset)]

    questions = train_qa['question'].values
    for q in questions:
        words = q.rstrip('?').split()
        for word in words:
            if len(word) >= 2:
                vocab[word] = vocab.get(word, 0) + 1
    vocab = Series(vocab)
    vocab.sort_values(ascending=False, inplace=True)
    vocab = DataFrame(vocab.iloc[0:3999])
    vocab.loc['<UNK>'] = [0]
    vocab.to_csv(vocab_path, columns=[], header=False)


def create_qa_encode(vttqa_path, vocab_path, answerset_path,
                     trainqa_encode_path, valqa_encode_path, testqa_encode_path):
    """Encode question/answer for generate batch faster.

    In train split, remove answers not in answer set and convert question and answer
    to one hot encoding. In val and test split, only convert question to one hot encoding.
    """
    train_qa = pd.read_json(os.path.join(vttqa_path, 'train_qa.json'))
    # remove question whose answer not in answer set
    answerset = pd.read_csv(answerset_path, header=None)[0]
    train_qa = train_qa[train_qa['answer'].isin(answerset)]

    val_qa = pd.read_json(os.path.join(vttqa_path, 'val_qa.json'))
    test_qa = pd.read_json(os.path.join(vttqa_path, 'test_qa.json'))
    vocab = pd.read_csv(vocab_path, header=None)[0]

    def _encode_question(row):
        """Map question to sequence of vocab id. 3999 for word not in vocab."""
        question = row['question']
        question_id = ''
        words = question.rstrip('?').split()
        for word in words:
            if word in vocab.values:
                question_id = question_id + \
                    str(vocab[vocab == word].index[0]) + ','
            else:
                question_id = question_id + '3999' + ','
        return question_id.rstrip(',')

    def _encode_answer(row):
        """Map answer to category id."""
        answer = row['answer']
        answer_id = answerset[answerset == answer].index[0]
        return answer_id

    print('start train split encoding.')
    train_qa['question_encode'] = train_qa.apply(_encode_question, axis=1)
    train_qa['answer_encode'] = train_qa.apply(_encode_answer, axis=1)
    print('start val split encoding.')
    val_qa['question_encode'] = val_qa.apply(_encode_question, axis=1)
    print('start test split encoding.')
    test_qa['question_encode'] = test_qa.apply(_encode_question, axis=1)

    train_qa.to_json(trainqa_encode_path, 'records')
    val_qa.to_json(valqa_encode_path, 'records')
    test_qa.to_json(testqa_encode_path, 'records')


def main():
    resFile_video = '/home/yulia/github/Activity_net_whole/train10_qa.json'
    data = json.load(open(resFile_video))
    caption_info = data['caption_info']

    max_num_captions_list = []
    for video_id in caption_info.keys():
        max_num_captions_list.append(len(caption_info[video_id]['timestamps']))

    max_num_captions = np.max(max_num_captions_list)
    max_num_frames = 16
    # print(max_num_captions) # for now maximum number of caps - 12

    # os.makedirs('data/msvd_qa')
    directory = '/home/yulia/github/Project_DVCVQA_ver1/helping/download_videos_train/'

    extract_video_feature(directory, max_num_captions, max_num_frames)

    # create_answerset(os.path.join(sys.argv[1], 'train_qa.json'),
    #                  'data/msvd_qa/answer_set.txt')
    #
    # create_vocab(os.path.join(sys.argv[1], 'train_qa.json'),
    #              'data/msvd_qa/answer_set.txt',
    #              'data/msvd_qa/vocab.txt')
    #
    # prune_embedding('data/msvd_qa/vocab.txt',
    #                 'util/glove.6B.300d.txt',
    #                 'data/msvd_qa/word_embedding.npy')
    #
    # create_qa_encode(sys.argv[1],
    #                  'data/msvd_qa/vocab.txt',
    #                  'data/msvd_qa/answer_set.txt',
    #                  'data/msvd_qa/train_qa_encode.json',
    #                  'data/msvd_qa/val_qa_encode.json',
    #                  'data/msvd_qa/test_qa_encode.json')


if __name__ == '__main__':
    main()
