import sys
sys.path.append('./external/coco/PythonAPI')
import os
import argparse
import numpy as np
import json
import skimage
sys.path.append('./external/refer')
from util import im_processing, text_processing
from util.io import load_referit_gt_mask as load_gt_mask
from refer import REFER
from pycocotools import mask as cocomask
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import shutil

def letterbox(img, mask, height, color=(123.7, 116.3, 103.5)):  # resize a rectangular image to a padded square
    shape = img.shape[:2]  # shape = [height, width]
    ratio = float(height) / max(shape)  # ratio  = old / new
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))
    dw = (height - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded square
    if mask is not None:
        mask = cv2.resize(mask, new_shape, interpolation=cv2.INTER_NEAREST)  # resized, no border
        mask = cv2.copyMakeBorder(mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)  # padded square
    return img, mask, ratio, dw, dh

def build_referit_batches(setname, T, input_H, input_W):
    # data directory
    im_dir = '/home/fg/Desktop/FG/refer/datasets/original_data/referitdata/images/' #original images
    mask_dir = '/home/fg/Desktop/FG/refer/datasets/original_data/referitdata/mask/' #original masks
    query_file = './data/referit/referit_query_' + setname + '.json'
    vocab_file = './data/vocabulary_referit.txt'

    # saving directory
    data_folder = '/home/fg/Desktop/FG/refer/datasets/referit/' + setname + '_batch/'
    data_prefix = 'referit_' + setname
    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    # load annotations
    query_dict = json.load(open(query_file))
    im_list = query_dict.keys()
    vocab_dict = text_processing.load_vocab_dict_from_file(vocab_file)

    # collect training samples
    samples = []
    for n_im, name in enumerate(im_list):
        im_name = name.split('_', 1)[0] + '.jpg'
        mask_name = name + '.mat'
        for sent in query_dict[name]:
            samples.append((im_name, mask_name, sent))

    # save batches to disk
    num_batch = len(samples)
    for n_batch in range(num_batch):
        print('saving batch %d / %d' % (n_batch + 1, num_batch))
        im_name, mask_name, sent = samples[n_batch]
        im = skimage.io.imread(im_dir + im_name)
        mask = load_gt_mask(mask_dir + mask_name).astype(np.float32)

        if 'train' in setname:
            im = skimage.img_as_ubyte(im_processing.resize_and_pad(im, input_H, input_W))
            mask = im_processing.resize_and_pad(mask, input_H, input_W)
        if im.ndim == 2:
            im = np.tile(im[:, :, np.newaxis], (1, 1, 3))

        text = text_processing.preprocess_sentence(sent, vocab_dict, T)

        np.savez(file = data_folder + data_prefix + '_' + str(n_batch) + '.npz',
            text_batch = text,
            im_batch = im,
            mask_batch = (mask > 0),
            sent_batch = [sent])


def build_coco_batches(dataset, setname, T):
    im_dir = '/media/8TB/sjy/refer'   #data directory
    im_type = 'train2014'
    vocab_file = './data/vocabulary_Gref.txt'

    # saving directory
    data_folder = '/media/8TB/sjy/refer/datasets/' + dataset + '/' + setname + '_batch/'
    data_prefix = dataset + '_' + setname
    if not os.path.isdir(data_folder + 'images/'):
        os.makedirs(data_folder + 'images/')
    if not os.path.isdir(data_folder + 'mask/'):
        os.makedirs(data_folder + 'mask/')

    if dataset == 'Gref':
        refer = REFER('./external/refer/data', dataset = 'refcocog', splitBy = 'google')
    elif dataset == 'unc':
        refer = REFER('./external/refer/data', dataset = 'refcoco', splitBy = 'unc')
    elif dataset == 'unc+':
        refer = REFER('./external/refer/data', dataset = 'refcoco+', splitBy = 'unc')
    else:
        raise ValueError('Unknown dataset %s' % dataset)
    refs = [refer.Refs[ref_id] for ref_id in refer.Refs if refer.Refs[ref_id]['split'] == setname]
    vocab_dict = text_processing.load_vocab_dict_from_file(vocab_file)

    Anns_list = []
    n_batch = 0
    for ref in tqdm(refs):
        im_name = 'COCO_' + im_type + '_' + str(ref['image_id']).zfill(12)
        im = skimage.io.imread('%s/%s/%s.jpg' % (im_dir, im_type, im_name))
        seg = refer.Anns[ref['ann_id']]['segmentation']
        box = refer.Anns[ref['ann_id']]['bbox'] #xmin, ymin, weight, height
        rle = cocomask.frPyObjects(seg, im.shape[0], im.shape[1])
        mask = np.max(cocomask.decode(rle), axis = 2).astype(np.float32)

        mask = skimage.img_as_ubyte(mask)

        for sentence in ref['sentences']:
            sent = sentence['sent']
            text = text_processing.preprocess_sentence(sent, vocab_dict, T)
            shutil.copyfile('%s/%s/%s.jpg' % (im_dir, im_type, im_name), '%s.jpg' % (data_folder + 'images/' + data_prefix + '_' + str(n_batch)))
            skimage.io.imsave('%s.png' % (data_folder + 'mask/' + data_prefix + '_' + str(n_batch)), mask)
            Anns_list.append(dict(file=data_prefix + '_' + str(n_batch),
                                  text_batch=text,
                                  sent_batch=[sent],
                                  bbox=box))
            n_batch += 1
    np.save(data_folder + data_prefix + '.npy', {'Anns': Anns_list})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type = str, default = 'unc') # 'unc', 'unc+', 'Grf'
    parser.add_argument('-t', type = str, default = 'train') # 'test', val', 'testA', 'testB'

    args = parser.parse_args()
    T = 20
    input_H = 320
    input_W = 320
    if args.d == 'referit':
        build_referit_batches(setname=args.t,
                              T=T, input_H=input_H, input_W=input_W)
    else:
        build_coco_batches(dataset=args.d, setname=args.t,
                           T=T, input_H=input_H, input_W=input_W)

