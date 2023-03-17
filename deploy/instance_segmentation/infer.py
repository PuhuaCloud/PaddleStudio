import cv2

from tools import *

if __name__ == '__main__':
    # 创建预测器
    model = init_predictor()
    
    # 读取图像
    img = cv2.imread('ps_imgpath')
    height, width = img.shape[:2]
    
    # 执行预测
    result = run(model, img)
    
    # nms优化
    resObj = result[0]
    indexList = NMS(resObj,0.5,0.3)
    print("检测到 " + str(sum(indexList)) + ' 个结果')
    
    # 展示最终结果
    for i in range(len(indexList)):
        if indexList[i]==0:
            continue
        xmin, ymin, xmax, ymax = int(resObj[i][2]),int(resObj[i][3]),int(resObj[i][4]),int(resObj[i][5])
        classNum = int(resObj[i][0])
        mask = (result[2][i] * 255).astype("uint8")
        mask = cv2.resize(mask, (width, height),cv2.INTER_NEAREST)   
        img = DrawInstanceResult(img,mask,classNum,xmin, ymin, xmax, ymax) 
    
    # 保存结果
    cv2.imwrite('result.jpg',img)