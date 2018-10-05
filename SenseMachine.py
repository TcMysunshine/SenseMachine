import numpy as np


class SenseMachine:
    '''learningRate是学习率，即是每次更新权重和截距的步长'''

    def __init__(self, learningRate=1):
        self.learningRare = learningRate
        self.w = 0
        self.b = 0

    def fit_transform(self, data, target):
        row = data.shape[0]
        col = data.shape[1]
        self.w = np.zeros([1, col])
        flag = True
        while flag:
            i = 0
            while i < row:
                '''对样本进行变换为列向量'''
                m = data[i].T
                # print(m)
                result = (np.dot(self.w, m) + self.b) * target[i]
                # print("计算之后的结果为：")
                # print(result)
                if result <= 0:
                    # print('分类错误')
                    self.w = self.w + target[i] * data[i] * self.learningRare
                    self.b = self.b + target[i] * self.learningRare
                    # print("调整之后的参数为：")
                    # print(self.w)
                    # print(self.b)
                    break
                else:
                    '''遍历结束且都能正确分类'''
                    if i == row - 1:
                        flag = False
                i = i + 1
        print("最终的参数")
        print(self.w)
        print(self.b)

    def predict(self, data):
        result = np.dot(self.w, data.T) + self.b
        print(result)
        if result > 0:
            return 1
        else:
            return -1

    '''获取Gram矩阵'''
    def getGramMatrix(self, data):
        row = data.shape[0]
        G = np.zeros([row, row])
        for i in range(row):
            for j in range(row):
                G[i][j] = np.dot(data[i], data[j])
        return G


if __name__ == "__main__":
    sm = SenseMachine(learningRate=1)
    data = np.array([[3, 3], [4, 3], [1, 1]])
    target = np.array([1, 1, -1])
    sm.fit_transform(data, target)
    predict_data = np.array([1, 1])
    print(sm.predict(predict_data))
    print(sm.getGramMatrix(data))
