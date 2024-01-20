import os
import numpy as np
import cv2
from time import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets #手写数据集要用到
from sklearn.manifold import TSNE
#该函数是关键，需要根据自己的数据加以修改，将图片存到一个np.array里面，并且制作标签
#因为是两类数据，所以我分别用0,1来表示
def get_data(Input_path): #Input_path为你自己原始数据存储路径，我的路径就是上的'D:\lei\B100'
    Image_names=os.listdir(Input_path) #获取目录下所有图片名称列表
    Image_names = Image_names[:10]
    data=np.zeros((len(Image_names),40000)) #初始化一个np.array数组用于存数据
    label=np.zeros((len(Image_names),)) #初始化一个np.array数组用于存数据
    #为前500个分配标签1，后500分配0
    for k in range(5):
        label[k]=1
    #读取并存储图片数据，原图为rgb三通道，而且大小不一，先灰度化，再resize成200x200固定大小
    for i in range(len(Image_names)):
        image_path=os.path.join(Input_path,Image_names[i])
        img=cv2.imread(image_path)
        img_gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        img=cv2.resize(img_gray,(200,200))
        img=img.reshape(1,40000)
        data[i]=img
        n_samples, n_features = data.shape
    return data, label, n_samples, n_features
 
# 下面的两个函数，
# 一个定义了二维数据，一个定义了3维数据的可视化
# 不作详解，也无需再修改感兴趣可以了解matplotlib的常见用法
# 
def plot_embedding_2D(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i]),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig
def plot_embedding_3D(data,label,title):
    x_min, x_max = np.min(data,axis=0), np.max(data,axis=0)
    data = (data- x_min) / (x_max - x_min)
    #ax = plt.figure().add_subplot(111,projection='3d')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(data.shape[0]):
        ax.text(data[i, 0], data[i, 1], data[i,2],str(label[i]), color=plt.cm.Set1(label[i]),fontdict={'weight': 'bold', 'size': 9})
    return ax
 
#主函数
def main():
    data, label, n_samples, n_features = get_data('sam_continue_learning/data/HRSOD-TE/imgs') #根据自己的路径合理更改
    print('Begining......') #时间会较长，所有处理完毕后给出finished提示
    tsne_2D = TSNE(n_components=2, init='pca', random_state=0) #调用TSNE
    result_2D = tsne_2D.fit_transform(data)
    tsne_3D = TSNE(n_components=3, init='pca', random_state=0)
    result_3D = tsne_3D.fit_transform(data)
    print('Finished......')
    #调用上面的两个函数进行可视化
    fig1 = plot_embedding_2D(result_2D, label,'t-SNE')
    plt.show()
    plt.savefig("tsne_2d.jpg")
    fig2 = plot_embedding_3D(result_3D, label,'t-SNE')
    plt.show()
    plt.savefig("tsne_3d.jpg")
if __name__ == '__main__':
    main()