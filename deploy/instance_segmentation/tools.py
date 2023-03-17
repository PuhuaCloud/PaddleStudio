import cv2
import numpy as np
from paddle.inference import Config
from paddle.inference import create_predictor


def init_predictor():
    '''配置预测器'''
    config = Config('inference_model/model.pdmodel',
                    'inference_model/model.pdiparams')
    config.enable_use_gpu(1024, 0)
    config.enable_memory_optim()
    predictor = create_predictor(config)
    return predictor


def run(predictor, img):
    '''单张图像预测'''
    im_info = {
        'scale_factor': np.array([1., 1.], dtype=np.float32),
        'im_shape': None,
        'image': None,
    }
    # 短边对齐
    ref_size = ps_short_size
    im_h, im_w, _ = img.shape
    if im_w >= im_h:
        scale_factor = ref_size * 1.0 / im_rh
        im_rh = ref_size
        im_rw = int(im_w * 1.0 / im_h * ref_size)
    elif im_w < im_h:
        scale_factor = ref_size * 1.0 / im_w
        im_rw = ref_size
        im_rh = int(im_h * 1.0 / im_w * ref_size)
    im_info['im_shape'] = np.array([[im_rh, im_rw]], dtype=np.float32)
    im_info['scale_factor'] = np.array([[scale_factor, scale_factor]],
                                       dtype=np.float32)
    # 32倍数截断
    im_rw = im_rw - im_rw % 32
    im_rh = im_rh - im_rh % 32
    img = cv2.resize(img, (im_rw, im_rh))
    # bgr转rgb && hwc转chw
    img = img[:, :, ::-1].astype('float32').transpose((2, 0, 1)) / 255
    # 归一化
    mean = [ps_mean0, ps_mean1, ps_mean2]
    std = [ps_std0, ps_std1, ps_std2]
    img_mean = np.array(mean).reshape((3, 1, 1))
    img_std = np.array(std).reshape((3, 1, 1))
    img -= img_mean
    img /= img_std
    img = img[np.newaxis, :]
    # 定义输入
    im_info['image'] = img
    input_names = predictor.get_input_names()
    for _, name in enumerate(input_names):
        input_tensor = predictor.get_input_handle(name)
        input_tensor.reshape(im_info[name].shape)
        input_tensor.copy_from_cpu(im_info[name].copy())
    # 推理
    predictor.run()
    # 获取结果
    results = []
    output_names = predictor.get_output_names()
    for _, name in enumerate(output_names):
        output_tensor = predictor.get_output_handle(name)
        output_data = output_tensor.copy_to_cpu()
        results.append(output_data)
    return results


def NMS(resObj, thrScore, thrIou):
    '''
    输入：resObj: [boxNum,6]     
    输出：indexList：有效框标识列表
    '''
    boxNum = resObj.shape[0]
    indexList = [1 for x in range(0, boxNum)]
    # 阈值检验
    for i in range(0, boxNum):
        if resObj[i][1] < thrScore:
            indexList[i] = 0
    # Iou检验
    for i in range(0, boxNum - 1):
        if indexList[i] == 0:
            continue
        for j in range(i+1, boxNum):
            if indexList[j] == 0:
                continue
            if indexList[i] == 0:
                continue
            classi = (int)(resObj[i][0])
            classj = (int)(resObj[j][0])
            if classi != classj:
                continue
            # 计算Iou

            pxmin, pymin, pxmax, pymax = resObj[i][2], resObj[i][3], resObj[i][4], resObj[i][5]
            gxmin, gymin, gxmax, gymax = resObj[j][2], resObj[j][3], resObj[j][4], resObj[j][5]

            parea = (pxmax - pxmin) * (pymax - pymin)  # 计算P的面积
            garea = (gxmax - gxmin) * (gymax - gymin)  # 计算G的面积

            # 求相交矩形的左下和右上顶点坐标(xmin, ymin, xmax, ymax)
            xmin = max(pxmin, gxmin)  # 得到左下顶点的横坐标
            ymin = max(pymin, gymin)  # 得到左下顶点的纵坐标
            xmax = min(pxmax, gxmax)  # 得到右上顶点的横坐标
            ymax = min(pymax, gymax)  # 得到右上顶点的纵坐标

            # 计算相交矩形的面积
            w = xmax - xmin
            h = ymax - ymin
            if w <= 0 or h <= 0:
                IoU = 0
            else:
                area = w * h  # G∩P的面积
                # 并集的面积 = 两个矩形面积 - 交集面积
                IoU = area / (parea + garea - area + 0.00000001)
            if IoU < thrIou:
                continue
            scorei = resObj[i][1]
            scorej = resObj[j][1]
            if scorei > scorej:
                indexList[j] = 0
            else:
                indexList[i] = 0
    return indexList


def DrawInstanceResult(img,mask,classNum,xmin, ymin, xmax, ymax):
    '''
    根据实例分割结果，在原图上展示结果
    '''
    height, width = img.shape[:2]
    # 画框
    default_font_scale = max(np.sqrt(height * width) // 900, .5)
    linewidth = max(default_font_scale / 40, 2)
    img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,0,255),linewidth)
    # 画掩码 
    image = img.astype('float32')
    alpha = .7
    w_ratio = .4
    color_mask = np.asarray( (255,0,0), dtype=int)
    for c in range(3):
        color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio * 255
    idx = np.nonzero(mask)
    image[idx[0], idx[1], :] *= 1.0 - alpha
    image[idx[0], idx[1], :] += alpha * color_mask
    img = image.astype("uint8")
    contours = cv2.findContours(
        mask, cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_NONE)[-2]
    img = cv2.drawContours(
        img,
        contours,
        contourIdx=-1,
        color=(0,255,0),
        thickness=1,
        lineType=cv2.LINE_AA)
    # 画类别号
    text_pos = (xmin, ymin)
    w= xmax- xmin
    h = ymax - ymin
    instance_area = w * h
    if (instance_area <100 or h < 40):
        if ymin >= height - 5:
            text_pos = (xmin, ymin)
        else:
            text_pos = (xmin, ymax)
    height_ratio = h / np.sqrt(height * width)
    font_scale = (np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2,
                            2) * 0.5 * default_font_scale)
    text = "class num: {}".format(str(classNum))
    (tw, th), baseline = cv2.getTextSize(
        text,
        fontFace=cv2.FONT_HERSHEY_DUPLEX,
        fontScale=font_scale,
        thickness=1)
    img = cv2.rectangle(
        img,
        text_pos, (text_pos[0] + tw, text_pos[1] + th + baseline),
        color=(255,255,255),
        thickness=-1)
    img = cv2.putText(
        img,
        text, (text_pos[0], text_pos[1] + th),
        fontFace=cv2.FONT_HERSHEY_DUPLEX,
        fontScale=font_scale,
        color=(0, 0, 255),
        thickness=1,
        lineType=cv2.LINE_AA)
    return img
        