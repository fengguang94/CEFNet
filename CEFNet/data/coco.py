import os
import os.path as osp
import sys
import torch
import torch.utils.data as data
import torch.nn.functional as F
import cv2
import numpy as np
from .config import cfg
from pycocotools import mask as maskUtils
import copy
from scipy.ndimage.morphology import distance_transform_edt


def get_label_map():
    if cfg.dataset.label_map is None:
        return {x + 1: x + 1 for x in range(len(cfg.dataset.class_names))}
    else:
        return cfg.dataset.label_map


class COCOAnnotationTransform(object):
    """Transforms a COCO annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    """

    def __init__(self):
        self.label_map = get_label_map()

    def __call__(self, target, width, height):
        """
        Args:
            target (dict): COCO target json annotation as a python dict
            height (int): height
            width (int): width
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class idx]
        """
        scale = np.array([width, height, width, height])
        res = []
        for obj in target:
            if 'bbox' in obj:
                bbox = obj['bbox']
                label_idx = self.label_map[obj['category_id']] - 1
                final_box = list(np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]) / scale)
                final_box.append(label_idx)
                res += [final_box]  # [xmin, ymin, xmax, ymax, label_idx]
            else:
                print("No bbox found for object ", obj)

        return res  # [[xmin, ymin, xmax, ymax, label_idx], ... ]


class ReferDataset(data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        set_name (string): Name of the specific set of COCO images.
        transform (callable, optional): A function/transform that augments the
                                        raw images`
        target_transform (callable, optional): A function/transform that takes
        in the target (bbox) and transforms it.
        prep_crowds (bool): Whether or not to prepare crowds for the evaluation step.
    """

    def __init__(self, image_path, info_file, resize_gt=True,
                 img_size=None,
                 dataset_name='MS COCO', has_gt=True):
        self.Anns = np.load(info_file, allow_pickle=True).item()['Anns']
        self.img_root = image_path
        self.img_list = os.listdir(image_path)
        self.img_size = img_size
        self.name = dataset_name
        self.has_gt = has_gt
        self.resize_gt = resize_gt

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, (target, masks, num_crowds)).
                   target is the object returned by ``coco.loadAnns``.
        """
        im, text_batch, gt, masks, h, w, num_crowds = self.pull_item(index)
        return im, text_batch, (gt, masks, num_crowds)

    def __len__(self):
        return len(self.Anns)

    def pull_item(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target, masks, height, width, crowd).
                   target is the object returned by ``coco.loadAnns``.
            Note that if no crowd annotations exist, crowd will be None
        """
        temp_dict = self.Anns[index]
        file_name = temp_dict['file']
        text_batch = temp_dict['text_batch']
        bbox = copy.deepcopy(temp_dict['bbox'])
        bbox[2], bbox[3] = bbox[0] + bbox[2], bbox[1] + bbox[3]  # xmin, ymin, xmax, ymax
        img = cv2.imread(osp.join(self.img_root + 'images/', file_name + '.jpg'), cv2.IMREAD_COLOR)
        masks = np.rint(
            cv2.imread(osp.join(self.img_root + 'mask/', file_name + '.png'), cv2.IMREAD_GRAYSCALE) / 255).astype(
            np.uint8)
        height, width = masks.shape
        if self.resize_gt:
            img, mask, ratio, dw, dh = letterbox(img, masks, self.img_size)
            bbox[0], bbox[2] = bbox[0] * ratio + dw, bbox[2] * ratio + dw
            bbox[1], bbox[3] = bbox[1] * ratio + dh, bbox[3] * ratio + dh
            #mask = mask_to_onehot(mask, 2)
            mask = np.reshape(mask, (-1, self.img_size, self.img_size))
            edge_map = mask_to_binary_edges(mask, 2, num_classes = 1)
            masks = np.concatenate([mask, edge_map], axis=0)
            target = [bbox[0], bbox[1], bbox[2], bbox[3], 1]
            target = np.array(target)
            #target = np.array(target) / self.img_size
            target = np.reshape(target, (1, -1))
        else:
            img, _, ratio, dw, dh = letterbox(img, None, self.img_size)
            # masks = np.reshape(masks, (-1, height, width))
            target = [bbox[0] / width, bbox[1] / height, bbox[2] / width, bbox[3] / height, 0]
            #target = [bbox[0], bbox[1], bbox[2], bbox[3], 1]
            target = np.array(target)
            target = np.reshape(target, (1, -1))

        # Separate out crowd annotations. These are annotations that signify a large crowd of
        # objects of said class, where there is no annotation for each individual object. Both
        # during testing and training, consider these crowds as neutral.
        num_crowds = 0
        # if max(bbox)>self.img_size:
        #     print(index)
        #     print(file_name)
        #     print(temp_dict['bbox'])
        #     print(height)
        #     print(width)
        # if max(target)>self.img_size:
        #     print(target)
        return torch.from_numpy(img).permute(2, 0, 1), text_batch, target, masks, height, width, num_crowds

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            cv2 img
        '''
        img_id = self.ids[index]
        path = self.coco.loadImgs(img_id)[0]['file_name']
        return cv2.imread(osp.join(self.root, path), cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        return self.coco.loadAnns(ann_ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def enforce_size(img, word_idx, targets, masks, num_crowds, new_w, new_h):
    """ Ensures that the image is the given size without distorting aspect ratio. """
    with torch.no_grad():
        _, h, w = img.size()

        if h == new_h and w == new_w:
            return img, word_idx, targets, masks, num_crowds

        # Resize the image so that it fits within new_w, new_h
        w_prime = new_w
        h_prime = h * new_w / w

        if h_prime > new_h:
            w_prime *= new_h / h_prime
            h_prime = new_h

        w_prime = int(w_prime)
        h_prime = int(h_prime)

        # Do all the resizing
        img = F.interpolate(img.unsqueeze(0), (h_prime, w_prime), mode='bilinear', align_corners=False)
        img.squeeze_(0)

        # Act like each object is a color channel
        masks = F.interpolate(masks.unsqueeze(0), (h_prime, w_prime), mode='bilinear', align_corners=False)
        masks.squeeze_(0)

        # Scale bounding boxes (this will put them in the top left corner in the case of padding)
        targets[:, [0, 2]] *= (w_prime / new_w)
        targets[:, [1, 3]] *= (h_prime / new_h)

        # Finally, pad everything to be the new_w, new_h
        pad_dims = (0, new_w - w_prime, 0, new_h - h_prime)
        img = F.pad(img, pad_dims, mode='constant', value=0)
        masks = F.pad(masks, pad_dims, mode='constant', value=0)

        return img, word_idx, targets, masks, num_crowds


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and (lists of annotations, masks)

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list<tensor>, list<tensor>, list<int>) annotations for a given image are stacked
                on 0 dim. The output gt is a tuple of annotations and masks.
    """
    targets = []
    word_idx = []
    imgs = []
    masks = []
    num_crowds = []

    for sample in batch:
        imgs.append(sample[0])
        word_idx.append(torch.FloatTensor(sample[1]).long())
        targets.append(torch.FloatTensor(sample[2][0]))
        masks.append(torch.FloatTensor(sample[2][1]))
        num_crowds.append(sample[2][2])

    return imgs, word_idx, (targets, masks, num_crowds)


def letterbox(img, mask, height, color=(0.0, 0.0, 0.0)):  # resize a rectangular image to a padded square
    #color=(123.7, 116.3, 103.5)
    shape = img.shape[:2]  # shape = [height, width]
    ratio = float(height) / max(shape)  # ratio  = old / new
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))
    dw = (height - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, (height, height), interpolation=cv2.INTER_AREA)  # resized, no border
    #img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    #img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded square
    if mask is not None:
        mask = cv2.resize(mask, (height, height), interpolation=cv2.INTER_NEAREST)  # resized, no border
        #mask = cv2.resize(mask, new_shape, interpolation=cv2.INTER_NEAREST)  # resized, no border
        #mask = cv2.copyMakeBorder(mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)  # padded square
    img = img.astype(np.float32)
    if mask is not None:
        mask = mask.astype(np.float32)
    return img, mask, ratio, dw, dh

def mask_to_binary_edges(mask, radius, num_classes):
    """
    Converts a segmentation mask (K,H,W) to a binary edgemap (H,W)
    """

    if radius < 0:
        return mask

    # We need to pad the borders for boundary conditions
    mask_pad = np.pad(mask, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)

    edgemap = np.zeros(mask.shape[1:])

    for i in range(num_classes):
        dist = distance_transform_edt(mask_pad[i, :]) + distance_transform_edt(1.0 - mask_pad[i, :])
        dist = dist[1:-1, 1:-1]
        dist[dist > radius] = 0
        edgemap += dist
    edgemap = np.expand_dims(edgemap, axis=0)
    edgemap = (edgemap > 0).astype(np.uint8)
    return edgemap

def mask_to_onehot(mask, num_classes):
    """
    Converts a segmentation mask (H,W) to (K,H,W) where the last dim is a one
    hot encoding vector
    """
    _mask = [mask == (i + 1) for i in range(num_classes)]
    return np.array(_mask).astype(np.uint8)
