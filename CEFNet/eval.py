# import os
# os.environ["CUDA_VISIBLE_DEVICES"]='1'
import cv2
from data import ReferDataset, get_label_map, MEANS  # , COLORS
from utils.functions import MovingAverage, ProgressBar
from utils import timer
from utils.functions import SavePath
from data import cfg, set_cfg, set_dataset
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import argparse
import random
import matplotlib.pyplot as plt
import torch.nn.functional as F
from model import model
import time


def seed_torch(seed=666):
    # random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True


seed_torch()


# # all boxes are [num, height, width] binary array
def compute_mask_IU(masks, target):
    assert (target.shape[-2:] == masks.shape[-2:])
    I = np.sum(np.logical_and(masks, target))
    U = np.sum(np.logical_or(masks, target))
    return I, U


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Evaluation')
    parser.add_argument('--trained_model',
                        default='weights/Base_0_50.pth', type=str,
                        help='Trained state_dict file path to open. If "interrupt", this will open the interrupt file.')
    parser.add_argument('--top_k', default=5, type=int,
                        help='Further restrict the number of predictions to parse')
    parser.add_argument('--cuda', default=True, type=str2bool,
                        help='Use cuda to evaulate model')
    parser.add_argument('--fast_nms', default=True, type=str2bool,
                        help='Whether to use a faster, but not entirely correct version of NMS.')
    parser.add_argument('--cross_class_nms', default=False, type=str2bool,
                        help='Whether compute NMS cross-class or per-class.')
    parser.add_argument('--display_masks', default=True, type=str2bool,
                        help='Whether or not to display masks over bounding boxes')
    parser.add_argument('--display_bboxes', default=True, type=str2bool,
                        help='Whether or not to display bboxes around masks')
    parser.add_argument('--display_text', default=True, type=str2bool,
                        help='Whether or not to display text (class [score])')
    parser.add_argument('--display_scores', default=True, type=str2bool,
                        help='Whether or not to display scores in addition to classes')
    parser.add_argument('--display', dest='display', action='store_true',
                        help='Display qualitative results instead of quantitative ones.')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true',
                        help='Shuffles the images when displaying them. Doesn\'t have much of an effect when display is off though.')
    parser.add_argument('--ap_data_file', default='results/ap_data.pkl', type=str,
                        help='In quantitative mode, the file to save detections before calculating mAP.')
    parser.add_argument('--resume', dest='resume', action='store_true',
                        help='If display not set, this resumes mAP calculations from the ap_data_file.')
    parser.add_argument('--max_images', default=-1, type=int,
                        help='The maximum number of images from the dataset to consider. Use -1 for all.')
    parser.add_argument('--output_coco_json', dest='output_coco_json', action='store_true',
                        help='If display is not set, instead of processing IoU values, this just dumps detections into the coco json file.')
    parser.add_argument('--bbox_det_file', default='results/bbox_detections.json', type=str,
                        help='The output file for coco bbox results if --coco_results is set.')
    parser.add_argument('--mask_det_file', default='results/mask_detections.json', type=str,
                        help='The output file for coco mask results if --coco_results is set.')
    parser.add_argument('--config', default='default_cfg',
                        help='The config object to use.')
    parser.add_argument('--output_web_json', dest='output_web_json', action='store_true',
                        help='If display is not set, instead of processing IoU values, this dumps detections for usage with the detections viewer web thingy.')
    parser.add_argument('--web_det_path', default='web/dets/', type=str,
                        help='If output_web_json is set, this is the path to dump detections into.')
    parser.add_argument('--no_bar', dest='no_bar', action='store_true',
                        help='Do not output the status bar. This is useful for when piping to a file.')
    parser.add_argument('--display_lincomb', default=False, type=str2bool,
                        help='If the config uses lincomb masks, output a visualization of how those masks are created.')
    parser.add_argument('--benchmark', default=False, dest='benchmark', action='store_true',
                        help='Equivalent to running display mode but without displaying an image.')
    parser.add_argument('--no_sort', default=False, dest='no_sort', action='store_true',
                        help='Do not sort images by hashed image ID.')
    parser.add_argument('--seed', default=None, type=int,
                        help='The seed to pass into random.seed. Note: this is only really for the shuffle and does not (I think) affect cuda stuff.')
    parser.add_argument('--mask_proto_debug', default=False, dest='mask_proto_debug', action='store_true',
                        help='Outputs stuff for scripts/compute_mask.py.')
    parser.add_argument('--no_crop', default=False, dest='crop', action='store_false',
                        help='Do not crop output masks with the predicted bounding box.')
    parser.add_argument('--image', default=None, type=str,
                        help='A path to an image to use for display.')
    parser.add_argument('--images', default=None, type=str,
                        help='An input folder of images and output folder to save detected images. Should be in the format input->output.')
    parser.add_argument('--video', default=None, type=str,
                        help='A path to a video to evaluate on. Passing in a number will use that index webcam.')
    parser.add_argument('--video_multiframe', default=1, type=int,
                        help='The number of frames to evaluate in parallel to make videos play at higher fps.')
    parser.add_argument('--score_threshold', default=0, type=float,
                        help='Detections with a score under this threshold will not be considered. This currently only works in display mode.')
    parser.add_argument('--dataset', default=None, type=str,
                        help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')
    parser.add_argument('--detect', default=False, dest='detect', action='store_true',
                        help='Don\'t evauluate the mask branch at all and only do object detection. This only works for --display and --benchmark.')
    parser.add_argument('--display_fps', default=False, dest='display_fps', action='store_true',
                        help='When displaying / saving video, draw the FPS on the frame')
    parser.add_argument('--emulate_playback', default=False, dest='emulate_playback', action='store_true',
                        help='When saving a video, emulate the framerate that you\'d get running in real-time mode.')

    parser.set_defaults(no_bar=False, display=False, resume=False, output_coco_json=False, output_web_json=False,
                        shuffle=False,
                        benchmark=False, no_sort=False, no_hash=False, mask_proto_debug=False, crop=True, detect=False,
                        display_fps=False,
                        emulate_playback=False)

    global args
    args = parser.parse_args(argv)

    if args.output_web_json:
        args.output_coco_json = True

    if args.seed is not None:
        random.seed(args.seed)


def evaluate(net: model, dataset, train_mode=False):
    dataset_size = len(dataset) if args.max_images < 0 else min(args.max_images, len(dataset))

    print()


    dataset_indices = list(range(len(dataset)))

    if args.shuffle:
        random.shuffle(dataset_indices)

    dataset_indices = dataset_indices[:dataset_size]
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    cum_I, cum_U = 0, 0
    seg_total = 0.
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    print(len(dataset_indices))

    #temp_Anns = np.load(cfg.valid_info, allow_pickle=True).item()['Anns']
    t_prediction = 0.0
    net(torch.zeros(1, 3, cfg.max_size, cfg.max_size).cuda(), torch.ones(1, 20).cuda().long())
    net(torch.zeros(1, 3, cfg.max_size, cfg.max_size).cuda(), torch.ones(1, 20).cuda().long())
    net(torch.zeros(1, 3, cfg.max_size, cfg.max_size).cuda(), torch.ones(1, 20).cuda().long())
    net(torch.zeros(1, 3, cfg.max_size, cfg.max_size).cuda(), torch.ones(1, 20).cuda().long())
    net(torch.zeros(1, 3, cfg.max_size, cfg.max_size).cuda(), torch.ones(1, 20).cuda().long())
    net(torch.zeros(1, 3, cfg.max_size, cfg.max_size).cuda(), torch.ones(1, 20).cuda().long())
    try:
        # Main eval loop
        for it, image_idx in enumerate(dataset_indices):
            timer.reset()

            img, text_batch, gt, gt_masks, h, w, num_crowd = dataset.pull_item(image_idx)

            batch = Variable(img.unsqueeze(0))
            if args.cuda:
                batch = batch.cuda()

            with timer.env('Network Extra'):
                start_time = time.time()
                preds = net(batch, torch.FloatTensor(text_batch).long().unsqueeze(0).cuda())
                diff_time = time.time() - start_time
                t_prediction += diff_time
                print('Detection took {}s per image'.format(diff_time))

            preds.gt_(1e-9)
            masks = F.interpolate(preds, (h, w), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)

            # temp_dict = temp_Anns[image_idx]
            # file_name = temp_dict['file']
            # sentences = temp_dict['sent_batch'][0]
            # root = "map/" + file_name + '(' + sentences + ')' + ".png"
            # cv2.imwrite(root, masks.cpu().numpy() * 255)

            I, U = compute_mask_IU(gt_masks, masks.cpu().numpy())
            cum_I += I
            cum_U += U
            for n_eval_iou in range(len(eval_seg_iou_list)):
                eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                seg_correct[n_eval_iou] += (I / U >= eval_seg_iou)
            seg_total += 1


        # Print results
        print('\nSegmentation evaluation (without DenseCRF):')
        result_str = ''
        for n_eval_iou in range(len(eval_seg_iou_list)):
            result_str += 'precision@%s = %f\n' % \
                          (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] / seg_total)
        result_str += 'overall IoU = %f\n' % (cum_I / cum_U)
        print(result_str)

        print("Total prediction time: {}. Average: {}/image. {}FPS".format(
            t_prediction, t_prediction / len(dataset_indices), 1.0/(t_prediction / len(dataset_indices))))


    except KeyboardInterrupt:
        print('Stopping...')
    return result_str


if __name__ == '__main__':
    parse_args()

    if args.config is not None:
        set_cfg(args.config)

    if args.trained_model == 'interrupt':
        args.trained_model = SavePath.get_interrupt('weights/')
    elif args.trained_model == 'latest':
        args.trained_model = SavePath.get_latest('weights/', cfg.name)

    if args.config is None:
        model_path = SavePath.from_str(args.trained_model)
        # TODO: Bad practice? Probably want to do a name lookup instead.
        args.config = model_path.model_name + '_config'
        print('Config not specified. Parsed %s from the file name.\n' % args.config)
        set_cfg(args.config)

    if args.detect:
        cfg.eval_mask_branch = False

    if args.dataset is not None:
        set_dataset(args.dataset)

    with torch.no_grad():
        if args.cuda:
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        if args.image is None and args.video is None and args.images is None:
            dataset = ReferDataset(image_path=cfg.valid_images,
                                   info_file=cfg.valid_info,
                                   img_size=cfg.max_size, resize_gt=False)
        else:
            dataset = None

        print('Loading model...', end='')
        net = model()
        net.load_weights(args.trained_model)
        net.eval()
        print(' Done.')

        if args.cuda:
            net = net.cuda()

        evaluate(net, dataset)
