"""Preprocess the data for model."""
import os
import inspect
import csv
import cv2
import numpy as np
from PIL import Image
import skvideo.io
import scipy
import tensorflow as tf
import pandas as pd

from .vgg16 import Vgg16
from .c3d import c3d
from moviepy.editor import *


class VideoVGGExtractor(object):
    """Select uniformly distributed frames and extract its VGG feature."""

    def __init__(self, frame_num, sess):
        """Load VGG model.

        Args:
            frame_num: number of frames per video.
            sess: tf.Session()
        """
        self.frame_num = frame_num
        self.inputs = tf.placeholder(tf.float32, [self.frame_num, 224, 224, 3])
        self.vgg16 = Vgg16()
        self.vgg16.build(self.inputs)
        self.sess = sess


    def _select_frames(self, path, list_timestamps, max_num_captions):
        """Select representative frames for video.
        Ignore some frames both at begin and end of video.
        Args:
            path: Path of video.
        Returns:
            frames: list of frames.
        """

        frames = list()
        video_data = skvideo.io.vread(path)
        total_frames = video_data.shape[0]

        for i in range(len(list_timestamps)):
            start = list_timestamps[i][0]
            end = list_timestamps[i][1]
            frame_point = (end - start) / 2
            #         print(start, end, frame_point)
            frame_data = video_data[int(frame_point)]
            img = Image.fromarray(frame_data)
            img = img.resize((224, 224), Image.BILINEAR)
            frame_data = np.array(img)
            frames.append(frame_data)
        if len(list_timestamps) < max_num_captions: #if number of captions of the current video is less than maximum number of captions among the whole dataset, just add fake frames
            for i in range(max_num_captions - len(list_timestamps)):
                img = Image.new('RGB', (224, 224), color = 'white')
                frame_data = np.array(img)
                frames.append(frame_data)

        return frames

    def extract(self, path,list_timestamps, max_num_captions):
        """Get VGG fc7 activations as representation for video.

        Args:
            path: Path of video.
        Returns:
            feature: [batch_size, 4096]
        """

        frames = self._select_frames(path, list_timestamps, max_num_captions)
        # We usually take features after the non-linearity, by convention.
        feature = self.sess.run(
            self.vgg16.relu7, feed_dict={self.inputs: frames})
        return feature


class VideoC3DExtractor(object):
    """Select uniformly distributed clips and extract its C3D feature."""

    def __init__(self, clip_num, max_num_captions, max_num_frames, sess):
        """Load C3D model."""
        self.clip_num = max_num_captions
        self.max_num_captions = max_num_captions
        self.max_num_frames = max_num_frames
        self.inputs = tf.placeholder(
            tf.float32, [self.max_num_captions, self.max_num_frames, 112, 112, 3])
        _, self.c3d_features = c3d(self.inputs, 1, max_num_captions)
        saver = tf.train.Saver()
        path = inspect.getfile(VideoC3DExtractor)
        path = os.path.abspath(os.path.join(path, os.pardir))
        saver.restore(sess, os.path.join(
            path, 'sports1m_finetuning_ucf101.model'))
        self.mean = np.load(os.path.join(path, 'crop_mean.npy'))
        self.sess = sess

    def _select_clips(self, path, list_timestamps):
        """Select self.batch_size clips for video. Each clip has 16 frames.

        Args:
            path: Path of video.
        Returns:
            clips: list of clips.

        """
        print('select clips')
        n = 0
        clips = list()
        video_data = skvideo.io.vread(path)
        total_frames = video_data.shape[0]
        for i in range(len(list_timestamps)):

            start_video = 0
            start = list_timestamps[i][0]
            end = list_timestamps[i][1]
            subclip_at_begining = VideoFileClip(path).subclip(start_video, start)
            num_frames_at_begining = len(list(subclip_at_begining.iter_frames()))

            if end > VideoFileClip(path).duration:
                end = VideoFileClip(path).duration
            subclip = VideoFileClip(path).subclip(start, end)
            num_frames = len(list(subclip.iter_frames()))

            if num_frames > self.max_num_frames:
                num_frames = self.max_num_frames  # if number of frames is more that max_num_frames -> cut it to max_num_frames

            if num_frames_at_begining + num_frames > total_frames:
                end_frame = total_frames
            else:
                end_frame = num_frames_at_begining + num_frames

            new_clip = []

            for k in range(num_frames_at_begining, end_frame):
                frame_data = video_data[int(k)]
                img = Image.fromarray(frame_data)
                img = img.resize((112, 112), Image.BILINEAR)
                frame_data = np.array(img) * 1.0
                # frame_data -= mean[j]
                new_clip.append(frame_data)
                n += 1
            if n < self.max_num_frames:  # if number of frames is less than max_num_frames, add fake ones to max_num_frames
                for l in range(self.max_num_frames - n):
                    img = Image.new('RGB', (112, 112), color='white')
                    frame_data = np.array(img)
                    new_clip.append(frame_data)
            #         print(len(new_clip))
            #         continue
            clips.append(new_clip)

        if len(list_timestamps) < self.max_num_captions:  # if number of slices in video is less than max_num_captions, make fake ones
            check = self.max_num_captions - len(list_timestamps)
            #         print(check)
            for j in range(check):
                new_clip = []
                for m in range(0, self.max_num_frames):
                    img = Image.new('RGB', (112, 112), color='white')
                    frame_data = np.array(img)
                    new_clip.append(frame_data)
                clips.append(new_clip)

        return clips


    def extract(self, path, list_timestamps):
        """Get 4096-dim activation as feature for video.

        Args:
            path: Path of video.
        Returns:
            feature: [self.batch_size, 4096]
        """
        clips = self._select_clips(path, list_timestamps)

        feature = self.sess.run(
            self.c3d_features, feed_dict={self.inputs: clips})
        return feature


def prune_embedding(vocab_path, glove_path, embedding_path):
    """Prune word embedding from pre-trained GloVe.

    For words not included in GloVe, set to average of found embeddings.

    Args:
        vocab_path: vocabulary path.
        glove_path: pre-trained GLoVe word embedding.
        embedding_path: .npy for vocabulary embedding.
    """
    # load GloVe embedding.
    glove = pd.read_csv(
        glove_path, sep=' ', quoting=csv.QUOTE_NONE, header=None)
    glove.set_index(0, inplace=True)
    # load vocabulary.
    vocab = pd.read_csv(vocab_path, header=None)[0]

    embedding = np.zeros([len(vocab), len(glove.columns)], np.float64)
    not_found = []
    for i in range(len(vocab)):
        word = vocab[i]
        if word in glove.index:

            embedding[i] = glove.loc[word]
        else:
            not_found.append(i)
    print('Not found:\n', vocab.iloc[not_found])

    embedding_avg = np.mean(embedding, 0)
    embedding[not_found] = embedding_avg

    np.save(embedding_path, embedding.astype(np.float32))
