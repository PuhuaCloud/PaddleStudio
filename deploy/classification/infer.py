import numpy as np
import cv2

from tools import *

if __name__ == '__main__':
    # 创建预测器
    model = init_predictor()
    # 读取图像
    img = cv2.imread('ps_imgpath')
    # 图像预处理
    img = preprocess(img)
    # 执行预测
    result = run(model, [img])
    label = np.argmax(result[0][0])
    score = result[0][0][label]
    print("class index: ", label ,"   score: ",score)