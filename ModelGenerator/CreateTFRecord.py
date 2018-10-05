import argparse
import io
import os
import sys
import threading
import numpy as np
import tensorflow as tf
from datetime import datetime
from queue import Queue
from PIL import Image


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _convert_to_example(image_buffer, image_width: int, image_height: int, filename: str, label: str,
                        class_index: float):
    example = tf.train.Example(features=tf.train.Features(feature={
        'image': _bytes_feature(image_buffer),
        'image_width': _int64_feature(image_width),
        'image_height': _int64_feature(image_height),
        "filename": _bytes_feature(filename.encode('utf8')),
        "label": _bytes_feature(label.encode('utf8')),
        "class_index": _int64_feature(class_index),
    }))
    return example


def _process_image_files_batch(thread_index, batch_data, shards, total_shards, output_dir,
                               output_name, error_queue):
    batch_size = len(batch_data)
    batch_per_shard = batch_size // len(shards)

    counter = 0
    error_counter = 0
    for s in range(len(shards)):
        shard = shards[s]
        output_filename = '%s-%.5d-of-%.5d' % (output_name, shard, total_shards)
        output_file = os.path.join(output_dir, output_filename)

        writer = tf.python_io.TFRecordWriter(output_file)
        shard_counter = 0
        shard_range = (s * batch_per_shard, min(batch_per_shard, batch_size - (s * batch_per_shard)))
        files_in_shard = np.arange(shard_range[0], shard_range[1], dtype=int)
        for i in files_in_shard:
            filename, class_index, label = data[i]
            try:
                image = Image.open(filename)
                with io.BytesIO() as image_buffer:
                    image.save(image_buffer, "png")
                    example = _convert_to_example(image_buffer.getvalue(), image.width, image.height, filename,
                                                  label, int(class_index))
                    writer.write(example.SerializeToString())
                shard_counter += 1
                counter += 1
            except StopIteration as e:
                error_counter += 1
                error_msg = repr(e)
                error_queue.put(error_msg)

        print('%s [thread %d]: Wrote %d images to %s, with %d errors.' %
              (datetime.now(), thread_index, shard_counter, output_file, error_counter))
        sys.stdout.flush()

    print('%s [thread %d]: Wrote %d images to %d shards, with %d errors.' %
          (datetime.now(), thread_index, counter, len(shards), error_counter))
    sys.stdout.flush()


def _create_tf_record(data, output_dir, output_name, num_shards, num_threads):
    num_data_per_thread = len(data) // num_threads
    num_shard_per_thread = num_shards // num_threads
    batch_data_ranges = [
        (i * num_data_per_thread, min((i + 1) * num_data_per_thread, len(data)))
        for i in range(num_threads)]

    coord = tf.train.Coordinator()

    error_queue = Queue()

    threads = []
    for thread_index in range(1, num_threads + 1):
        batch_ranges = batch_data_ranges[thread_index - 1]
        batch_data = data[batch_ranges[0]:batch_ranges[1]]
        shards = [thread_index + (thread_index - 1) * (num_shard_per_thread - 1) + shard
                  for shard in range(num_shard_per_thread)]
        args = (thread_index, batch_data, shards, num_shards, output_dir, output_name, error_queue)
        t = threading.Thread(target=_process_image_files_batch, args=args)
        t.start()
        threads.append(t)

    coord.join(threads)

    errors = []
    while not error_queue.empty():
        errors.append(error_queue.get())
    print('%d examples failed.' % (len(errors),))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--score_image_dir', help='Directory of the score images.', type=str, default="data/scores")
    parser.add_argument('--other_image_dir', help='Directory of the other images.', type=str, default="data/other")
    parser.add_argument('--train_size', help='Ratio of training samples.', type=float, default=0.6)
    parser.add_argument('--validation_size', help='Ratio of validation samples.', type=float, default=0.2)
    parser.add_argument('--output_dir', help='Directory for the tfrecords.', type=str, default="data")
    parser.add_argument('--shuffle', help='Shuffle the samples.', action='store_true', default=True)
    parser.add_argument('--shards', help='Number of shards to make.', type=int, default=8)
    parser.add_argument('--threads', help='Number of threads to use.', type=int, default=8)
    parsed_args = parser.parse_args()
    return parsed_args


def assert_args(args):
    assert os.path.exists(args.score_image_dir), "Images directory does not exist"
    assert os.path.exists(args.other_image_dir), "Images directory does not exist"
    assert args.train_size + args.validation_size <= 1, "Train ratio + validation ratio must be <= 1"
    assert args.train_size > 0, "Train ratio must be > 0"
    assert args.validation_size >= 0, "Validation ratio must be >= 0"
    assert args.shards > 0, "Number of shards must be > 0"
    assert args.threads > 0, "Number of threads must be > 0"
    assert args.shards % args.threads == 0


if __name__ == '__main__':
    args = parse_args()
    assert_args(args)

    score_images = [os.path.join(args.score_image_dir, filename) for filename in os.listdir(args.score_image_dir)]
    other_images = [os.path.join(args.other_image_dir, filename) for filename in os.listdir(args.other_image_dir)]

    labels = np.concatenate((["score"] * len(score_images), ["other"] * len(other_images)))
    classes = np.concatenate(([0] * len(score_images), [1] * len(other_images)))
    images = np.concatenate((score_images, other_images))
    data = np.column_stack((images, classes, labels))

    if args.shuffle:
        np.random.shuffle(data)

    num_data = len(data)
    num_train = round(num_data * args.train_size)
    num_validation = round(num_data * args.validation_size)

    training = data[:num_train]
    _create_tf_record(training, args.output_dir, "train", args.shards, args.threads)
    if args.validation_size > 0:
        validation = data[num_train:num_train + num_validation]
        _create_tf_record(validation, args.output_dir, "validation", args.shards, args.threads)
    if args.train_size + args.validation_size < 1:
        test = data[num_train + num_validation:]
        _create_tf_record(test, args.output_dir, "test", args.shards, args.threads)
