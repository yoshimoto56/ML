import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

DIM = 3

#3次元プロット
def plot(data, label, surf=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = ("red", "blue")
    groups = ("normal", "diseased") 
    if (surf!=None):
        X, Y, Z = surf
        ax.plot_wireframe(X, Y, Z, color="black")
    temp = data[np.where(label == 0)]
    x,y,z = temp.T
    ax.scatter(x, y, z, alpha=1, c=colors[0], edgecolors='none', s=30, label=groups[0])
    temp = data[np.where(label == 1)]
    x,y,z = temp.T
    ax.scatter(x, y, z, alpha=1, c=colors[1], edgecolors='none', s=30, label=groups[1])
    plt.legend(loc=2)
    plt.show()

#テスト用3次元データ
def genData(num, filename):
    cov = [[3,1,1],[1,3,1],[1,1,3]]#共分散
    x_a = np.random.multivariate_normal([-5,-5,-5], cov, num)
    x_b = np.random.multivariate_normal([5,5,5], cov, num)
    feature = np.r_[x_a, x_b]
    #ラベルデータ
    l_a = np.full((num), 0)
    l_b = np.full((num), 1)
    label = np.r_[l_a, l_b]
    #結合して保存
    data = np.c_[feature, label]
    np.savetxt(filename,data, delimiter=',')
    #確認用プロット
    if(DIM == 3):
       plot(feature, label)#qで終了

#データの読み込み
def loadData(filename):
    data = np.loadtxt(filename, delimiter=',')
    x = np.delete(data, (data.shape[1]-1), 1)
    data = data.transpose()
    y = data[(data.shape[0]-1),]
    return x,y

#LDAによる学習
def training(x,y):
    lda = LDA(n_components=3)#[TODO:次元削減，引数無しで元の特徴量次元]
    lda.fit(x, y)
    print("Training accuracy = ", lda.score(x,y))
    print("W = ", lda.coef_)
    print("b = ", lda.intercept_)
    label = lda.predict(x)#推定結果

    #プロットして確認．色つけは推定値に基づく
    if(DIM == 3):
        #識別に用いる超平面の生成
        xt = np.arange(-10, 10, 1)
        yt = np.arange(-10, 10, 1)
        xt, yt = np.meshgrid(xt, yt)
        zt = (- lda.intercept_[0] - xt * lda.coef_[0,0] - yt *lda.coef_[0,1])/lda.coef_[0,2]
        surf = (xt,yt,zt)   
        data = x
        plot(data, label, surf)
    return lda

#LDAによる評価
def testing(model, x,y):
    print("Testing accuracy = ", model.score(x,y))
    label = model.predict(x)#推定結果
    return label

if __name__ == '__main__':
    genData(50, 'data/data.csv')#ファイル名data.csvで学習用データ生成
    genData(50, 'data/test.csv')#ファイル名test.csvで評価用データ生成
    #データの読み込み
    x_training,y_training = loadData("data/data.csv")
    x_testing,y_testing = loadData("data/test.csv")
    #学習
    model = training(x_training, y_training)
    #評価
    testing(model, x_testing, y_testing)

