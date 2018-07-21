import tensorflow as tf
import argparse
import glob
import os
from PIL import Image
import csv
from capsnet import MultiSegCaps
from data import make_dataset
from loss import weighted_margin_loss, reconstruction_loss

class_weight = [0.2595, 0.1826, 4.5640, 0.1417, 0.9051, 0.3826, 9.6446, 1.8418, 0.6823, 6.2478, 7.3614, 0.]
# class_weight = [0, 1, 1]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['segcaps', 'unet', 'segnet'],
                        help="Model: you can select from segcaps, unet, segnet")
    parser.add_argument('--n_class', type=int, default=12,
                        help="Number of segmentation class")
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="Learning rate: default is 1e-3")
    parser.add_argument('--batch_size', type=int, default=1,
                        help="Batch size: default is 1")
    parser.add_argument('--epoch', type=int, default=200,
                        help="Epoch: default is 200")
    parser.add_argument('--val_step', type=int, default=10,
                        help="Step of validation: default is 10")
    parser.add_argument('--rs', type=float, default=0.005,
                        help="Scale of reconstruction loss: default is 0.0005")
    parser.add_argument('--img_channel', type=int, default=3,
                        help="Channel of input images: default 3")
    args = parser.parse_args()
    return args


def train(args):
    tf.enable_eager_execution()
    tf.executing_eagerly()
    tfe = tf.contrib.eager
    writer = tf.contrib.summary.create_file_writer("./log")
    global_step = tf.train.get_or_create_global_step()
    writer.set_as_default()
    dataset = make_dataset("./train.csv", args.batch_size, args.n_class)
    model = MultiSegCaps(n_class=args.n_class)
    #optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=args.lr)
    checkpoint_dir = './models'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    root = tfe.Checkpoint(optimizer=optimizer,
                          model=model,
                          optimizer_step=tf.train.get_or_create_global_step())


    with tf.contrib.summary.record_summaries_every_n_global_steps(50):
        for epoch in range(args.epoch):
            for imgs, lbls in dataset:
                global_step.assign_add(1)
                with tf.GradientTape() as tape:
                    out_seg, reconstruct = model(imgs, lbls)

                    segmentation_loss = tf.losses.softmax_cross_entropy(lbls, out_seg)
                    tf.contrib.summary.scalar('segmentation_loss', segmentation_loss)
                    #segmetation_loss = weighted_margin_loss(out_seg, lbls, class_weighting=[0,1,1,1,1])

                    reconstruct_loss = reconstruction_loss(reconstruct, imgs, rs=args.rs)
                    tf.contrib.summary.scalar('reconstruction_loss', reconstruct_loss)

                    total_loss = segmentation_loss + reconstruct_loss
                    tf.contrib.summary.scalar('total_loss', total_loss)
                    print(total_loss)
                grad = tape.gradient(total_loss, model.variables)
                optimizer.apply_gradients(zip(grad, model.variables),
                                          global_step=tf.train.get_or_create_global_step())

            if epoch % 10 == 0:
                root.save(file_prefix=checkpoint_prefix)




if __name__ == "__main__":
    args = get_args()
    train(args)

