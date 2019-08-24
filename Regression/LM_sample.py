import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model as lm

#function: 時刻同期用関数
#objctive: 線形補間により配列Aの時刻に配列Bの時刻を合わせる
#input: 配列A, 配列Bの順．いずれも最初の列が時刻．
#outpt: 戻り値は同期後のa,bのデータ
def sync(data_a, data_b):
    cj=1
    nrow = data_a.shape[0]
    ncol = (data_b.shape[1]-1)
    data_a_sync = data_a[0:,1:]
    #配列を確保
    data_b_sync = np.zeros((nrow, ncol))
    for i, a in enumerate(data_a):
        for j in range(cj, data_b.shape[0]):
            if a[0]<=data_b[j,0]:
                #線形補間
                t1 = (data_b[j,0] - a[0]) / (data_b[j,0] - data_b[j-1,0])
                t2 = (a[0] - data_b[j-1,0]) / (data_b[j,0] - data_b[j-1,0])
                data_b_sync[i,] = data_b[j-1,1:] * t1 + data_b[j,1:] * t2
                cj = j
                break
    return data_a_sync, data_b_sync

#function: プロット保存関数
#objctive: xとyの関係を重ねて描画した結果を保存する
#input: 1次元x,多次元y,yの次元,xラベル,yラベル,x上下限度,y上下限度,ファイル名
#outpt: なし
def savefig(x, ys, dim, xlabel, ylabel, xlim, ylim, filename):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    cm = plt.get_cmap('jet',dim)
    if dim>1 :
        for i in range(0,dim):
            ax.plot(x, ys[0:,i], linestyle='-', color=cm(i), label='chn '+str(i))
    else:
        ax.plot(x, ys, linestyle='-', color=cm(0))
    
    plt.xlim(xlim)
    ax.set_xlabel(xlabel)
    plt.ylim(ylim)
    ax.set_ylabel(ylabel)
    if dim>1:
        ax.legend(loc='best')
    plt.savefig(filename)
    
#function: メイン関数
#objective: 多入力1出力の線形回帰を行う
def main():
    #データの読み込み
    data_in = np.loadtxt('data/sample_in.csv', delimiter=',')
    data_out = np.loadtxt('data/sample_out.csv', delimiter=',')

    #時系列のデータを保存
    tlim = [0,1]        #時刻の上下限
    input_ylim = [0,1]  #inputの上下限
    output_ylim = [0,3] #outputの上下限
    savefig(data_in[0:,0],data_in[0:,1:],data_in.shape[1]-1,'Time [s]', 'Voltage [V]', tlim, input_ylim, 'data/time-input.png')
    savefig(data_out[0:,0],data_out[0:,1],1,'Time [s]', 'Displacement [mm]', tlim, output_ylim, 'data/time-out.png')

    #データの時刻を揃える
    data_in_sync,data_out_sync = sync(data_in, data_out)
    np.savetxt('data/sample_in_sync.csv', data_in_sync, delimiter=',')
    np.savetxt('data/sample_out_sync.csv', data_out_sync, delimiter=',')

    #各チャンネルの入力と出力関係をプロット
    for i in range(0,data_in_sync.shape[1]):
        savefig(data_in_sync[0:,i],data_out_sync[0:,0],1,'Voltage [V]', 'Displacement [mm]', input_ylim, output_ylim, 'data/input'+str(i)+'-output.png')       
  
    #重回帰分析
    clf = lm.LinearRegression() 
    X = data_in_sync
    Y = data_out_sync[0:,0]
    clf.fit(X, Y)
    Y_est = clf.predict(X)
    Y_bind =np.column_stack([Y,Y_est])
    RMSE = np.sqrt(np.dot((Y_est-Y),(Y_est-Y))/Y.shape[0])

    #結果の値を表示・推定結果のグラフを保存
    print('W=')
    print(clf.coef_)
    print('b=')
    print(clf.intercept_)
    print('R^2=')
    print(clf.score(X, Y))
    print('RMSE=')
    print(RMSE)
    print('NRMSE=')
    print(RMSE/max(abs(Y)))
    savefig(data_in[0:,0],Y_bind,2,'Time [s]', 'Displacement [mm]', tlim, output_ylim, 'data/time-result.png')

if __name__ == '__main__':
    main()