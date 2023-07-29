import mmcv
import numpy as np
import cv2
from mmcv.image import imread, imwrite
# from mmrotate.core.visualization.image import imshow
from mmcv.visualization.color import color_val
import math
import random

def imshow(img, win_name='', wait_time=0):
    """Show an image.
    Args:
        img (str or ndarray): The image to be displayed.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
    """
    cv2.imshow(win_name, imread(img))
    if wait_time == 0:  # prevent from hanging if windows was closed
        while True:
            ret = cv2.waitKey(1)

            closed = cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1
            # if user closed window or if some key pressed
            if closed or ret != -1:
                break
    else:
        ret = cv2.waitKey(wait_time)

def imshow_det_rbboxes(
    img,
    bboxes,
    labels,
    class_names=None,
    score_thr=0.3,
    bbox_color=(226, 43, 138),
    text_color="white",
    thickness=2,
    font_scale=0.25,
    show=True,
    win_name="",
    wait_time=0,
    out_file=None,
    random_color=True,
):
    """Draw bboxes and class labels (with scores) on an image.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 5) or
            (n, 6).
        labels (ndarray): Labels of bboxes.
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown.
        bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
        text_color (str or tuple or :obj:`Color`): Color of texts.
        thickness (int): Thickness of lines.
        font_scale (float): Font scales of texts.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str or None): The filename to write the image.
    """
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    img = imread(img)

    scores = bboxes[:, -1]
    inds = scores > score_thr
    bboxes = bboxes[inds, :]

    bbox_color = (226, 43, 138) if bbox_color is None else color_val(bbox_color)
    text_color = (255, 255, 255) if text_color is None else color_val(text_color)
    for bbox, label in zip(bboxes, labels):
        xc, yc, w, h, ag, score = bbox.tolist()
        wx, wy = w / 2 * math.cos(ag), w / 2 * math.sin(ag)
        hx, hy = -h / 2 * math.sin(ag), h / 2 * math.cos(ag)
        p1 = (xc - wx - hx, yc - wy - hy)
        p2 = (xc + wx - hx, yc + wy - hy)
        p3 = (xc + wx + hx, yc + wy + hy)
        p4 = (xc - wx + hx, yc - wy + hy)
        ps = np.int0(np.array([p1, p2, p3, p4]))
        if not random_color:
            cv2.drawContours(img, [ps], -1, bbox_color, thickness=thickness)
        else:
            cv2.drawContours(
                img,
                [ps],
                -1,
                [random.randint(0, 255) for _ in range(3)],
                thickness=thickness,
            )
        # label_text = class_names[
        #     label] if class_names is not None else 'cls {}'.format(label)
        # label_text += '|{:.02f}'.format(score)
        # font = cv2.FONT_HERSHEY_COMPLEX
        # text_size = cv2.getTextSize(label_text, font, font_scale, 1)
        # text_width = text_size[0][0]
        # text_height = text_size[0][1]
        # cv2.rectangle(img, (int(xc), int(yc) - text_height - 2),
        #               (int(xc) + text_width, int(yc) + 3), (0, 128, 0), -1)
        # cv2.putText(img, label_text, (int(xc), int(yc)), font, font_scale,
        #             text_color, 1)

    if show:
        imshow(img, win_name, wait_time)
    if out_file is not None:
        imwrite(img, out_file)

    return img


def imshow_det_bboxes(
    img,
    bboxes,
    labels,
    class_names=None,
    score_thr=0.3,
    bbox_color=(226, 43, 138),
    text_color="white",
    thickness=2,
    font_scale=0.25,
    show=True,
    win_name="",
    wait_time=0,
    out_file=None,
    random_color=True,
):
    """Draw bboxes and class labels (with scores) on an image.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 5) or
            (n, 6).
        labels (ndarray): Labels of bboxes.
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown.
        bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
        text_color (str or tuple or :obj:`Color`): Color of texts.
        thickness (int): Thickness of lines.
        font_scale (float): Font scales of texts.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str or None): The filename to write the image.
    """
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    img = imread(img)

    scores = bboxes[:, -1]
    inds = scores > score_thr
    bboxes = bboxes[inds, :]

    bbox_color = (226, 43, 138) if bbox_color is None else color_val(bbox_color)
    text_color = (255, 255, 255) if text_color is None else color_val(text_color)
    for bbox, label in zip(bboxes, labels):
        x1, y1, x2, y2, score = bbox.tolist()

        p1 = (x1, y1)
        p2 = (x1, y2)
        p3 = (x2, y2)
        p4 = (x2, y1)
        ps = np.int0(np.array([p1, p2, p3, p4]))
        if not random_color:
            cv2.drawContours(img, [ps], -1, bbox_color, thickness=thickness)
        else:
            cv2.drawContours(
                img,
                [ps],
                -1,
                [random.randint(0, 255) for _ in range(3)],
                thickness=thickness,
            )
        # label_text = class_names[
        #     label] if class_names is not None else 'cls {}'.format(label)
        # label_text += '|{:.02f}'.format(score)
        # font = cv2.FONT_HERSHEY_COMPLEX
        # text_size = cv2.getTextSize(label_text, font, font_scale, 1)
        # text_width = text_size[0][0]
        # text_height = text_size[0][1]
        # cv2.rectangle(img, (int(xc), int(yc) - text_height - 2),
        #               (int(xc) + text_width, int(yc) + 3), (0, 128, 0), -1)
        # cv2.putText(img, label_text, (int(xc), int(yc)), font, font_scale,
        #             text_color, 1)

    if show:
        imshow(img, win_name, wait_time)
    if out_file is not None:
        imwrite(img, out_file)

    return img


def show_result(
    img,
    result,
    score_thr=0.3,
    bbox_color=(226, 43, 138),
    text_color="white",
    thickness=2,
    font_scale=0.25,
    win_name="",
    show=False,
    wait_time=0,
    out_file=None,
    classes=None,
    rotate=True,
    **kwargs
):

    img = mmcv.imread(img)
    img = img.copy()
    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    # draw segmentation masks
    if segm_result is not None and len(labels) > 0:
        segms = mmcv.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        np.random.seed(42)
        color_masks = [
            np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            for _ in range(max(labels) + 1)
        ]
        for i in inds:
            i = int(i)
            color_mask = color_masks[labels[i]]
            mask = segms[i]
            img[mask] = img[mask] * 0.5 + color_mask * 0.5
    # if out_file specified, do not show image in window
    if out_file is not None:
        show = False
    # draw bounding boxes
    if rotate:
        imshow_det_rbboxes(
            img,
            bboxes,
            labels,
            class_names=classes,
            score_thr=score_thr,
            bbox_color=bbox_color,
            text_color=text_color,
            thickness=thickness,
            font_scale=font_scale,
            win_name=win_name,
            show=show,
            wait_time=wait_time,
            out_file=out_file,
        )
    else:
        imshow_det_bboxes(
            img,
            bboxes,
            labels,
            class_names=classes,
            score_thr=score_thr,
            bbox_color=bbox_color,
            text_color=text_color,
            thickness=thickness,
            font_scale=font_scale,
            win_name=win_name,
            show=show,
            wait_time=wait_time,
            out_file=out_file,
        )

    if not (show or out_file):
        return img
