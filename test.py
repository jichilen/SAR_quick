import numpy as np
import torch

def bbox_overlaps(bboxes1, bboxes2, mode='iou', allow_neg=False):
    """Calculate the ious between each bbox of bboxes1 and bboxes2.

    Args:
        bboxes1(ndarray): shape (n, 4)
        bboxes2(ndarray): shape (k, 4)
        mode(str): iou (intersection over union) or iof (intersection
            over foreground)

    Returns:
        ious(ndarray): shape (n, k)
    """

    assert mode in ['iou', 'iof']

    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = np.zeros((cols, rows), dtype=np.float32)
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
        bboxes1[:, 3] - bboxes1[:, 1] + 1)
    area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
        bboxes2[:, 3] - bboxes2[:, 1] + 1)
    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        if not allow_neg:
            overlap = np.maximum(x_end - x_start + 1, 0) * np.maximum(y_end - y_start + 1, 0)
        else:
            overlap = (x_end - x_start + 1) * (y_end - y_start + 1)
            flag = np.ones(overlap.shape)
            flag[x_end - x_start + 1 < 0] = -1.
            flag[y_end - y_start + 1 < 0] = -1.
            overlap = flag * np.abs(overlap)

        if mode == 'iou':
            union = area1[i] + area2 - overlap
        else:
            union = area1[i] if not exchange else area2
        ious[i, :] = overlap / union
    if exchange:
        ious = ious.T
    return ious

def boxlist_niou(boxlist1, boxlist2):
    # target, proposal
    """Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Arguments:
      boxlist1: (BoxList) bounding boxes, sized [N,4].
      boxlist2: (BoxList) bounding boxes, sized [M,4].

    Returns:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    if boxlist1.size != boxlist2.size:
        raise RuntimeError(
            "boxlists should have same image size, got {}, {}".format(boxlist1, boxlist2))

    # N = len(boxlist1)
    # M = len(boxlist2)

    area1 = boxlist1.area()
    area2 = boxlist2.area()

    box1, box2 = boxlist1.bbox, boxlist2.bbox

    in_lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    in_rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    TO_REMOVE = 1

    in_wh = (in_rb - in_lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
    inter = in_wh[:, :, 0] * in_wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    niou = (inter * 4 - 2 * area2 - area1[:, None]) / (area1[:, None] + area2 - inter)
    # giou = iou - (outer - area1[:, None] - area2 + inter) / outer
    return iou, niou

def bbox_overlaps_giou(bboxes1, bboxes2):
    """Calculate the gious between each bbox of bboxes1 and bboxes2.

    Args:
        bboxes1(ndarray): shape (n, 4)
        bboxes2(ndarray): shape (k, 4)

    Returns:
        gious(ndarray): shape (n, k)
    """


    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = np.zeros((cols, rows), dtype=np.float32)
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
        bboxes1[:, 3] - bboxes1[:, 1] + 1)
    area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
        bboxes2[:, 3] - bboxes2[:, 1] + 1)
    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        x_min = np.minimum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        y_min = np.minimum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        x_max = np.maximum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        y_max = np.maximum(bboxes1[i, 3], bboxes2[:, 3])

        overlap = np.maximum(x_end - x_start + 1, 0) * np.maximum(y_end - y_start + 1, 0)
        closure = np.maximum(x_max - x_min + 1, 0) * np.maximum(y_max - y_min + 1, 0)

        union = area1[i] + area2 - overlap
        print(overlap.dtype)
        print(union.dtype)
        print(closure)
        ious[i, :] = overlap / union - (closure - overlap) / closure
        print((closure - overlap))
    if exchange:
        ious = ious.T
    return ious

if __name__ == '__main__':
    print(bbox_overlaps_giou(np.array([[0,0,100,100]]*3), np.array([[0,200,100,300]]*3)))