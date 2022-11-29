# coding:utf-8
# date:2019-08
# Author: Eric.Lee
# function: predict camera
import os
import torch
import time
from utils.datasets import *
from utils.utils import *
from utils.parse_config import parse_data_cfg
from yolov3 import Yolov3, Yolov3Tiny
from utils.torch_utils import select_device
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"

img_size = 416  # 图像尺寸
conf_thres = 0.5  # 检测置信度
nms_thres = 0.6  # nms 阈值


def process_data(img, img_size=416):  # 图像预处理
    img, _, _, _ = letterbox(img, height=img_size)
    # Normalize RGB
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    return img


def show_model_param(model):
    params = list(model.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        # print("该层的结构: {}, 参数和: {}".format(str(list(i.size())), str(l)))
        k = k + l
    # print("----------------------")
    # print("总参数数量和: " + str(k))


def refine_hand_bbox(bbox, img_shape):
    height, width, _ = img_shape

    x1, y1, x2, y2 = bbox

    expand_w = (x2-x1)
    expand_h = (y2-y1)

    x1 -= expand_w*0.06
    y1 -= expand_h*0.1
    x2 += expand_w*0.06
    y2 += expand_h*0.1

    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    x1 = int(max(0, x1))
    y1 = int(max(0, y1))
    x2 = int(min(x2, width-1))
    y2 = int(min(y2, height-1))

    return (x1, y1, x2, y2)


def detect(file, base):
    # classes = load_classes(parse_data_cfg(data_cfg)['names'])

    # if not file:
    #     print('file empty')
    #     return False
    t = time.time()
    model = base['model']
    classes = base['classes']
    num_classes = len(classes)
    device = base['device']

    use_cuda = torch.cuda.is_available()
    colors = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32)
              for v in range(1, num_classes + 1)][::-1]

    loc_time = time.localtime()
    str_time = time.strftime("%Y-%m-%d_%H-%M-%S", loc_time)
    # im0 = cv2.imread(file)
    im0 = file
    img = process_data(im0, img_size)

    if use_cuda:
        torch.cuda.synchronize()
    t1 = time.time()
    # print("process time:", t1-t)
    img = torch.from_numpy(img).unsqueeze(0).to(device)
    pred, _ = model(img)  # 图片检测
    if use_cuda:
        torch.cuda.synchronize()
    t2 = time.time()
    # print("inference time:", t2-t1)
    detections = non_max_suppression(pred, conf_thres, nms_thres)[0]  # nms
    if use_cuda:
        torch.cuda.synchronize()
    t3 = time.time()
    # print("get res time:", t3-t2)

    if detections is None or len(detections) == 0:
        return False

     # Rescale boxes from 416 to true image size
    detections[:, :4] = scale_coords(
        img_size, detections[:, :4], im0.shape).round()

    result = []
    for res in detections:
        result.append((classes[int(res[-1])], float(res[4]),
                       [int(res[0]), int(res[1]), int(res[2]), int(res[3])]))
        if use_cuda:
            torch.cuda.synchronize()

    # for r in result:
        # print('识别结果------')
        # print(r)

    for *xyxy, conf, cls_conf, cls in detections:
        # label = '%s %.2f' % (classes[int(cls)], conf)
        # label = '%s' % (classes[int(cls)])

        # print(conf, cls_conf)
        # xyxy = refine_hand_bbox(xyxy,im0.shape)
        # react = int(xyxy[0]), int(xyxy[1])+6, int(xyxy[2]), int(xyxy[3])
        # 转数组
        react = [int(xyxy[0]), int(xyxy[1])+6, int(xyxy[2]), int(xyxy[3])]
        # 只要一只手
        break
        # if int(cls) == 0:
        #     plot_one_box(xyxy, im0, label=label, color=(
        #         15, 255, 95), line_thickness=3)
        # else:
        #     plot_one_box(xyxy, im0, label=label, color=(
        #         15, 155, 255), line_thickness=3)
    s2 = time.time()
    # print("detect time: {} \n".format(s2 - t))
    return react


def initModel(type='hand'):
    if type == 'hand':
        # 手
        voc_config = './cfg/hand.data'  # 模型相关配置文件
        model_path = './weights/hand/hand_416.pt'  # 检测模型路径
    model_cfg = 'yolo'  # yolo / yolo-tiny 模型结构

    # with torch.no_grad():#设置无梯度运行模型推理
    classes = load_classes(parse_data_cfg(voc_config)['names'])
    num_classes = len(classes)

    # Initialize model
    weights = model_path
    if "-tiny" in model_cfg:
        a_scalse = 416./img_size
        anchors = [(10, 14), (23, 27), (37, 58),
                   (81, 82), (135, 169), (344, 319)]
        anchors_new = [(int(anchors[j][0]/a_scalse), int(anchors[j][1]/a_scalse))
                       for j in range(len(anchors))]
        model = Yolov3Tiny(num_classes, anchors=anchors_new)

    else:
        a_scalse = 416./img_size
        anchors = [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                   (59, 119), (116, 90), (156, 198), (373, 326)]
        anchors_new = [(int(anchors[j][0]/a_scalse), int(anchors[j][1]/a_scalse))
                       for j in range(len(anchors))]
        model = Yolov3(num_classes, anchors=anchors_new)

    show_model_param(model)  # 显示模型参数

    device = select_device()  # 运行硬件选择
    # Load weights
    if os.access(weights, os.F_OK):  # 判断模型文件是否存在
        model.load_state_dict(torch.load(
            weights, map_location=device)['model'])
    else:
        print('error model not exists')
        return False
    model.to(device).eval()  # 模型模式设置为 eval
    return {'model': model, 'classes': classes, 'device': device}


def predict(file, model):
    with torch.no_grad():  # 设置无梯度运行模型推理
        return detect(
            file,
            model
        )


# if __name__ == '__main__':

    # 左右手
    # voc_config = 'cfg/helmet.data' # 模型相关配置文件
    # model_path = './weights/latest_416-20220101.pt' # 检测模型路径
    # 手
    # voc_config = 'cfg/hand.data'  # 模型相关配置文件
    # model_path = './weights/hand_416.pt'  # 检测模型路径

    # model_cfg = 'yolo'  # yolo / yolo-tiny 模型结构
    # video_path = "./video/left_right.mp4"  # 测试视频

    # img_size = 416  # 图像尺寸
    # conf_thres = 0.5  # 检测置信度
    # nms_thres = 0.6  # nms 阈值

    # with torch.no_grad():  # 设置无梯度运行模型推理
    #     detect(
    #         model_path=model_path,
    #         cfg=model_cfg,
    #         data_cfg=voc_config,
    #         img_size=img_size,
    #         conf_thres=conf_thres,
    #         nms_thres=nms_thres,
    #         video_path=video_path,
    #     )
