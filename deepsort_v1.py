import cv2
import numpy as np
import serial
import time
import wiringpi as wp
from wiringpi import GPIO
from deep_sort_realtime.deepsort_tracker import DeepSort  # 导入DeepSORT


# 初始化GPIO
GPIO_PIN = 7
wp.wiringPiSetup()
wp.pinMode(GPIO_PIN, wp.INPUT)

# 初始化串口
serial_port1 = '/dev/ttyS0'
baud_rate = 115200
ser1 = serial.Serial(serial_port1, baud_rate, timeout=0)

# 初始化DeepSORT
tracker = DeepSort(max_age=10, n_init=3)  # max_age: 目标消失后保留的帧数

# 常量定义
OBJ_THRESH = 0.82
NMS_THRESH = 0.01
IMG_SIZE = 640

CLASSES = ("recyclables", "kitchen", "harmful", "else")

# 全局变量
center_x = 640
center_y = 480
class_id = 6

def int_to_bytes(value, length, byte_order='big'):
    return int(value).to_bytes(length, byte_order)


def sender(posx, posy, class_type):
    data_x = int_to_bytes(int(posx), 2, 'big')
    data_y = int_to_bytes(int(posy), 2, 'big')
    class_id = int_to_bytes(int(class_type), 1, 'little')
    frame_head = b'\x21\x2c'
    frame_foot = b'\x5b'
    data_pack = frame_head + class_id + data_x + data_y + frame_foot
    ser1.write(data_pack)


def draw(image, boxes, scores, classes, ratio, padding, tracks):
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()  # 获取目标框的左上右下坐标

        # 计算中心点
        global center_x, center_y, class_id
        center_x = int((ltrb[0] + ltrb[2]) / 2)
        center_y = int((ltrb[1] + ltrb[3]) / 2)

        # 根据类别设置class_id
        if classes[track.det_class] == 0:
            class_id = 2
        elif classes[track.det_class] == 2:
            class_id = 4
        elif classes[track.det_class] == 3:
            class_id = 5
        elif classes[track.det_class] == 1:
            class_id = 3

        # 发送数据
        sender(center_x, center_y, class_id)
        print("发送成功:", center_x, center_y, class_id)

        # 绘制目标框和ID
        cv2.rectangle(image, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), (255, 0, 0), 2)
        cv2.putText(image, f'ID: {track_id} {CLASSES[classes[track.det_class]]} {scores[track.det_class]:.2f}',
                    (int(ltrb[0]), int(ltrb[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with object threshold.
    """
    box_confidences = box_confidences.reshape(-1)
    candidate, class_num = box_class_probs.shape

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    _class_pos = np.where(class_max_score * box_confidences >= OBJ_THRESH)
    scores = (class_max_score * box_confidences)[_class_pos]

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]

    return boxes, classes, scores


def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.
    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def dfl(position):
    # Distribution Focal Loss (DFL)
    # x = np.array(position)
    n, c, h, w = position.shape
    p_num = 4
    mc = c // p_num
    y = position.reshape(n, p_num, mc, h, w)

    # Vectorized softmax
    e_y = np.exp(y - np.max(y, axis=2, keepdims=True))  # subtract max for numerical stability
    y = e_y / np.sum(e_y, axis=2, keepdims=True)

    acc_metrix = np.arange(mc).reshape(1, 1, mc, 1, 1)
    y = (y * acc_metrix).sum(2)
    return y


def box_process(position):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE // grid_h, IMG_SIZE // grid_w]).reshape(1, 2, 1, 1)

    position = dfl(position)
    box_xy = grid + 0.5 - position[:, 0:2, :, :]
    box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
    xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)

    return xyxy


def yolov8_post_process(input_data):
    boxes, scores, classes_conf = [], [], []
    defualt_branch = 3
    pair_per_branch = len(input_data) // defualt_branch
    # Python 忽略 score_sum 输出
    for i in range(defualt_branch):
        boxes.append(box_process(input_data[pair_per_branch * i]))
        classes_conf.append(input_data[pair_per_branch * i + 1])
        scores.append(np.ones_like(input_data[pair_per_branch * i + 1][:, :1, :, :], dtype=np.float32))

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0, 2, 3, 1)
        return _in.reshape(-1, ch)

    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]

    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)

    # filter according to threshold
    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)

    # nms
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s)

        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores


def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
             new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=color)  # add border
    # return im
    return im, ratio, (left, top)





def myFunc(rknn_lite, IMG):
    IMG2 = cv2.cvtColor(IMG, cv2.COLOR_BGR2RGB)
    IMG2, ratio, padding = letterbox(IMG2)
    IMG2 = np.expand_dims(IMG2, 0)

    # YOLO推理
    outputs = rknn_lite.inference(inputs=[IMG2], data_format=['nhwc'])
    boxes, classes, scores = yolov8_post_process(outputs)

    if wp.digitalRead(GPIO_PIN) == wp.HIGH:
        if boxes is not None:
            # 将YOLO检测结果转换为DeepSORT输入格式
            detections = []
            for i in range(len(boxes)):
                box = boxes[i]
                class_id = classes[i]
                score = scores[i]
                detections.append(([box[0], box[1], box[2], box[3]], score, class_id))

            # 更新DeepSORT跟踪器
            tracks = tracker.update_tracks(detections, frame=IMG)

            # 绘制跟踪结果
            draw(IMG, boxes, scores, classes, ratio, padding, tracks)
    elif wp.digitalRead(GPIO_PIN) == wp.LOW:
        center_x = 640
        center_y = 480
        class_id = 6
        sender(center_x, center_y, class_id)
        print("暂休状态....")

    return IMG