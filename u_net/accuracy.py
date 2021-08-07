import numpy as np
from os.path import join
from PIL import Image

__all__ = ['SegmentationMetric']

"""
confusionMetric  # 注意：此处横着代表预测值，竖着代表真实值，与之前介绍的相反
P\L     P    N
P      TP    FP
N      FN    TN
"""


class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)

    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)  # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
        return meanAcc  # 返回单个值，如：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 =  0.89

    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)  # 取对角元素的值，返回列表
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        IoU = intersection / union  # 返回列表，其值为各个类别的IoU
        mIoU = np.nanmean(IoU)  # 求各类别IoU的平均
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel):  # 同FCN中score.py的fast_hist()函数
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
            np.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))

def myarray(gt_list, pred_list, ind):
    gt_imgs = [join(x) for x in gt_list]
    pred_imgs = [join(x) for x in pred_list]
    # for ind in range(len(gt_imgs)):


    pred = np.array(Image.open(pred_imgs[ind]))
    label = np.array(Image.open(gt_imgs[ind]))
    # for i in range(len(pred)):
    #     if pred[i] != 0:
    #         pred[i] = 1
    #
    # for i in range(len(label)):
    #     if label[i] != 0:
    #         label[i] = 1
    imgLabel = label.flatten()
    imgPredict = pred.flatten()
    return imgPredict, imgLabel


if __name__ == '__main__':
    gt_list = open("predict/gt.txt",'r').read().splitlines()
    pred_list = open("predict/pred.txt", 'r').read().splitlines()
    mymiou = 0
    mypa=0
    my_mpa=0
    for i in range(len(gt_list)):
        imgPredict, imgLabel = myarray(gt_list, pred_list, i)
        # print("imgPredict", imgPredict)
        for i in range(len(imgPredict)):
            if imgPredict[i] != 0:
                imgPredict[i] = 1

        for i in range(len(imgLabel)):
            if imgLabel[i] != 0:
                imgLabel[i] = 1
        metric = SegmentationMetric(2)  # 2表示有2个分类，有几个分类就填几
        metric.addBatch(imgPredict, imgLabel)
        pa = metric.pixelAccuracy()
        cpa = metric.classPixelAccuracy()
        mpa = metric.meanPixelAccuracy()
        mIoU = metric.meanIntersectionOverUnion()
        mymiou += mIoU
        mypa+=pa
        my_mpa+=mpa
        print('pa is : %f' % pa)
        # print(cpa)
        # print('cpa is :')  # 列表
        # print('mpa is : %f' % mpa)
        # print('IoU is : %f' % mIoU)

    print('acc is : %f' % (mypa/len(gt_list)))
    # print('miou is : %f' % (mymiou / len(gt_list)))
    # print('mpa is : %f' % (my_mpa / len(gt_list)))
