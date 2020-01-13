"""
对位置类别属性的数据集中的每个点一次执行以下操作：
1.计算已知类别数据集中的点与当前点之间的距离；
2.按照距离递增次序排序；
3.选取与当前点距离最小的k个点；
4.确定前k个点所在类别中的出现频率；
5.返回前k个点出现频率最高的类别作为当前点的预测分类。
"""
import datetime
import operator
import time
from os import listdir

from numpy import tile, array, zeros, shape
import matplotlib
import matplotlib.pyplot as plt

#----------------------------示例1：约会网站-------------------------
def createDataSet():
    group = array([[1.0, 1.1],
                   [1.0, 1.0],
                   [0.0, 0.0],
                   [0.0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    # 得到文件行数
    numberOfLines = len(arrayOLines)
    # 创建返回矩阵
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0 : 3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

def autoNorm(dataSet):
    """
    :param dataSet: 需要进行归一化的数据集
    :return: 归一化的数据集，范围矩阵，最小值矩阵
    """
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet/tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

def datingClassTest():
    """
    对约会网站的测试方法
    :return: 错误数
    """
    # 设置测试数据的的一个比例（训练数据集比例=1-hoRatio）
    hoRatio = 0.1  # 测试范围,一部分测试一部分作为样本
    # 从文件中加载数据
    datingDataMat, datingLabels = file2matrix('./data/datingTestSet2.txt')  # load data setfrom file
    # 归一化数据
    normMat, ranges, minVals = autoNorm(datingDataMat)
    # m 表示数据的行数，即矩阵的第一维
    m = normMat.shape[0]
    # 设置测试的样本数量， numTestVecs:m表示训练样本的数量
    numTestVecs = int(m * hoRatio)
    print('numTestVecs=', numTestVecs)
    errorCount = 0.0
    for i in range(numTestVecs):
        # 对数据测试
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))
    print(errorCount)

def classify0(inX, dataSet, labels, k):
    """
    :param inX: 需要进行分类的输入向量
    :param dataSet: 输入训练样本集
    :param labels: 标签向量
    :param k: 最近邻居数目
    :return:
    """
    dataSetSize = dataSet.shape[0]

    # 欧式距离计算
     # #前面用tile，把一行inX变成4行一模一样的
    # (tile有重复的功能，dataSetSize是重复4遍，
    # 后面的1保证重复完了是4行，而不是一行里有四个一样的)，
    # 然后再减去dataSet，是为了求两点的距离，先要坐标相减，这个就是坐标相减
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet  # 差矩阵
    sqDiffMat = diffMat ** 2  # square 平方差矩阵
    sqDistances = sqDiffMat.sum(axis=1)  #axis=1是求每行之和，这样得到了(x1-x2)^2+(y1-y2)^2
    distances = sqDistances ** 0.5

    # argsort是排序，将元素按照由小到大的顺序返回下标，比如([3,1,2]),它返回的就是([1,2,0])
    sortedDistIndicies = distances.argsort()
    classCount = {} #字典

    # 选取最小的k个点
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        # get是取字典里的元素，如果之前这个voteIlabel是有的，
        # 那么就返回字典里这个voteIlabel里的值，如果没有就返回0（后面一个参数），
        # 这行代码的意思就是算离目标点距离最近的k个点的类别，这个点是哪个类别哪个类别就加1
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # key=operator.itemgetter(1)的意思是按照字典里的第一个排序，
    # {A:1,B:2},要按照第1个（AB是第0个），即‘1’‘2’排序。reverse=True是降序排序
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

#----------------------------示例2：手写识别系统-------------------------
def img2vector(filename):
    """
    将图像数据转换为向量
    :param filename: 图片文件 因为我们的输入数据的图片格式是 32 * 32的二进制图像矩阵
    :return: 一维矩阵
    该函数将图像转换为向量：该函数创建 1 * 1024 的NumPy数组，然后打开给定的文件，
    循环读出文件的前32行，并将每行的头32个字符值存储在NumPy数组中，最后返回数组。
    """
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    # 1. 导入数据
    hwLabels = []
    trainingFileList = listdir('./data/trainingDigits')  # load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    # hwLabels存储0～9对应的index位置， trainingMat存放的每个位置对应的图片向量
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        # 将 32*32的矩阵->1*1024的矩阵
        trainingMat[i, :] = img2vector('./data/trainingDigits/%s' % fileNameStr)

    # 2. 导入测试数据
    testFileList = listdir('./data/testDigits')  # iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('./data/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount / float(mTest)))

if __name__ == '__main__':
    startTime = time.clock()
    handwritingClassTest()
    endTime = time.clock()
    print(endTime - startTime)
    # datingClassTest()
    # group, labels = createDataSet()
    # result = classify0([1, 2], group, labels, 3)
    # print(result)

"""
    datingDataMat, datingLabels = file2matrix("./data/datingTestSet2.txt")
    # result = classify0([14489,7.1534,1.673904], Mat, V, 5)
    # print(result)
    normMat, ranges, minVals = autoNorm(datingDataMat)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
    # ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
    ax.scatter(normMat[:, 0], normMat[:, 1], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
    plt.show()
    print(minVals, ranges)
"""
