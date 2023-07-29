# Copyright (c) OpenMMLab. All rights reserved.
import os.path
import os.path as osp
import pickle
import shutil
import tempfile
import time

import mmcv
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info
import cv2
from mmdet.core import encode_mask_results
from kcr.utils.rotate import obb2poly_np_le90
import numpy as np
import copy
from kcr.utils.rotate import poly2obb_np_le90_seg
# from bryce_tools.angle_from_mask import mmcv_test_replacer


def rotate_by_fg_seg(result, filename):
    img = cv2.imread(filename)
    name = filename.split('/')[-1].replace('.png', '')
    plot_dir = "/storage/alan/workspace/mmStorage/KCR/kcr_aa_hrsc_shiponly/foreground"
    boxes = result[0][0]
    polys = obb2poly_np_le90(boxes[:, :5]).reshape(-1, 4, 2)
    obbs = []
    for i, poly in enumerate(polys):
        pad = 100
        xmin = max(int(poly[:, 0].min()), 0)
        ymin = max(int(poly[:, 1].min()), 0)
        xmax = min(int(poly[:, 0].max()), img.shape[1])
        ymax = min(int(poly[:, 1].max()), img.shape[0])

        xmin_pad = max(xmin - pad, 0)
        ymin_pad = max(ymin - pad, 0)
        xmax_pad = min(xmax + pad, img.shape[1])
        ymax_pad = min(ymax + pad, img.shape[0])

        patch = img[ymin_pad: ymax_pad, xmin_pad: xmax_pad]
        foreground = copy.copy(img)
        mask = np.zeros(foreground.shape[:2], np.uint8)

        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        # rect = (xmin - xmin_pad, ymin - ymin_pad, foreground.shape[1] - (xmax_pad - xmax) - (xmin - xmin_pad), foreground.shape[0] - (ymax_pad - ymax) - (ymin - ymin_pad))
        rect = (xmin, ymin, xmax - xmin, ymax - ymin)
        cv2.grabCut(foreground, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        contours, _ = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # obb = []
        # for cnt in contours:
        #     obb.append(poly2obb_np_le90_seg(cnt))
        #
        # obb.sort(key=lambda x: x[2] * x[3])
        # if len(obb) == 0 or obb[-1][2] * obb[-1][3] < 50:
        #     obbs.append(boxes[i][:5])
        # else:
        #     obb = list(obb[+1])
        #     if obb[2] * obb[3] > boxes[i][2] * boxes[i][3]:
        #         obb[2] = boxes[i][2]
        #         obb[3] = boxes[i][3]
        #     obb[:1] = boxes[i][:1]
        #     shortside_obb = min(obb[2:4])
        #     shortside_aa = min(boxes[i][2:4])
        #     theta = abs(obb[4])
        #     theta = min(theta, 2 * np.pi - theta)
        #     theta = min(theta, np.pi - theta)
        #     theta = min(theta, np.pi/2 - theta)
        #     if shortside_obb > shortside_aa * np.cos(theta):
        #         if obb[2] > obb[3]:
        #             obb[3] = shortside_aa * np.cos(theta)
        #         else:
        #             obb[2] = shortside_aa * np.cos(theta)
        #
        #
        #     obbs.append(obb)
        #
        # foreground[mask2 > 0] = foreground[mask2 > 0] * 0.5 + np.array([[255, 0, 0]]) * 0.5
        # # # cv2.imwrite(osp.join(plot_dir, name + str(i) + '.png'), np.concatenate([patch, foreground], 1))
        # cv2.imwrite(osp.join(plot_dir, name + str(i) + '.png'), foreground)
    try:
        obbs = np.concatenate([np.array(obbs), boxes[:, -1:]], -1, dtype=np.float32)
        result[0][0] = obbs
    except:
        pass
    return result


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    model.eval()
    results = []
    dataset = data_loader.dataset
    PALETTE = getattr(dataset, 'PALETTE', None)
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        # USE opencv to rectify the result to make it rotated
        # result = mmcv_test_replacer(result, data['img_metas'][0].data[0][0]['filename'])

        batch_size = len(result)
        if show or out_dir:
            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None
                # color = (60, 220, 20)
                color = (220, 60, 20)
                model.module.show_result(
                    img_show,
                    result[i],
                    thickness=10,
                    font_size=15,
                    bbox_color=color,
                    text_color=color,
                    mask_color=color,
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr)

        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        # This logic is only used in panoptic segmentation test.
        elif isinstance(result[0], dict) and 'ins_results' in result[0]:
            for j in range(len(result)):
                bbox_results, mask_results = result[j]['ins_results']
                result[j]['ins_results'] = (bbox_results,
                                            encode_mask_results(mask_results))

        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            # encode mask results
            if isinstance(result[0], tuple):
                result = [(bbox_results, encode_mask_results(mask_results))
                          for bbox_results, mask_results in result]
            # This logic is only used in panoptic segmentation test.
            elif isinstance(result[0], dict) and 'ins_results' in result[0]:
                for j in range(len(result)):
                    bbox_results, mask_results = result[j]['ins_results']
                    result[j]['ins_results'] = (
                        bbox_results, encode_mask_results(mask_results))

        results.extend(result)

        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
