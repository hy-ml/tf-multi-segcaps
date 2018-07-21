import os
import glob
import argparse
import numpy as np


def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, required=True)
    parser.add_argument('--lbl_dir', type=str, required=True)
    parser.add_argument('--test_num', type=int, required=True,
                        help="The number of test data")
    parser.add_argument('--val_num', type=int, required=True,
                        help="The number of validation data")
    parser.add_argument('--lbl_extension', type=str, default="same",
                        help="Extension of label image: default is same, which means same "
                             "extension to input image")
    parser.add_argument('--output_dir', type=str, default=".")
    args = parser.parse_args()
    return args


def main(args):
    f_train = open(os.path.join(args.output_dir, "train.csv"), "w")
    f_val = open(os.path.join(args.output_dir, "val.csv"), "w")
    f_test = open(os.path.join(args.output_dir, "test.csv"), "w")

    img_list = glob.glob(os.path.join(args.img_dir, "*"))

    if args.lbl_extension == 'same':
        lbl_extension = img_list[0][-3:]
    else:
        lbl_extension = args.lbl_extension

    for i in range(args.test_num):
        index = np.random.randint(len(img_list))
        path_img = img_list[index]
        path_lbl = os.path.join(args.lbl_dir,
                                os.path.basename(path_img)[:-3] + lbl_extension)
        img_list.pop(i)
        f_test.write("{},{}\n".format(path_img, path_lbl))

    for i in range(args.val_num):
        index = np.random.randint(len(img_list))
        path_img = img_list[index]
        path_lbl = os.path.join(args.lbl_dir,
                                os.path.basename(path_img)[:-3] + lbl_extension)
        img_list.pop(i)
        f_val.write("{},{}\n".format(path_img, path_lbl))

    for path_img in img_list:
        path_lbl = os.path.join(args.lbl_dir,
                                os.path.basename(path_img)[:-3] + lbl_extension)
        f_train.write("{},{}\n".format(path_img, path_lbl))

    f_train.close()
    f_test.close()
    f_val.close()
    print("Success")


if __name__ == "__main__":
    args = get_argparse()
    main(args)



