import tensorflow as tf
import csv
import os


def make_dataset(dataset_txt, batch_size=1, n_class=2):

    def _img_preprocess(img):
        img = img * tf.constant([1/255], dtype=tf.float32)
        return img

    def _lbl_preprocess(lbl, n_class):
        lbl_one_hot = tf.one_hot(lbl, depth=n_class)
        return lbl_one_hot

    def _parse_function(img_name, lbl_name):
        img_string = tf.read_file(img_name)
        img_decoded = tf.image.decode_png(img_string)
        img_decoded = tf.cast(img_decoded, dtype=tf.float32)

        img_decoded_preprocessed = _img_preprocess(img_decoded)

        lbl_string = tf.read_file(lbl_name)
        lbl_decoded = tf.image.decode_png(lbl_string)
        lbl_decoded = tf.squeeze(lbl_decoded, axis=-1)

        lbl_decoded_preprocessed = _lbl_preprocess(lbl_decoded, n_class)

        return img_decoded_preprocessed, lbl_decoded_preprocessed

    # Start main processing here
    if not os.path.isfile(dataset_txt):
        print("There is not such file: {}".format(dataset_txt))
        exit(1)

    img_list = []
    lbl_list = []

    with open(dataset_txt, "r") as f:
        reader = csv.reader(f)

        for row in reader:
            img_list.append(row[0])
            lbl_list.append(row[1])

    dataset = tf.data.Dataset.from_tensor_slices((tf.constant(img_list), tf.constant(lbl_list)))
    dataset = dataset.map(_parse_function).shuffle(1000).batch(batch_size)

    return dataset