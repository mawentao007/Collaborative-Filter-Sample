#coding:utf-8
__author__ = 'marvin'

import pandas as pd
from scipy.spatial.distance import cosine

def loadData(dataName):
    return pd.read_csv(dataName)

def itemBasedCF(data):

#移除‘user’列，第一个参数表示要删除的列的label，第二个参数表示label所在的行号
    data_germany = data.drop('user',1)

#行索引是歌曲名，列也是，放i，j列的cos距离
    data_ibs = pd.DataFrame(index=data_germany.columns,columns=data_germany.columns)

    for i in range(0,len(data_ibs.columns)):
        for j in range(0,len(data_ibs.columns)):
            #1-第i列和第j列的cos距离
            data_ibs.ix[i,j] = 1-cosine(data_germany.ix[:,i],data_germany.ix[:,j])

#行索引为歌曲名，存储10列最相近的，索引1-10
    data_neighbours = pd.DataFrame(index=data_ibs.columns,columns=[range(1,11)])


    for i in range(0,len(data_ibs.columns)):
        #取ib中cos距离最近的10个元素的索引,也就是相似度最高的十首歌曲
        data_neighbours.ix[i,:10] = data_ibs.ix[:,i].order(ascending=False)[:10].index

    return data_neighbours


def userBasedCF(data,data_neighbours):

# Have an Item Based similarity matrix at your disposal (we do…wohoo!)
# Check which items the user has consumed
# For each item the user has consumed, get the top X neighbours
# Get the consumption record of the user for each neighbour.
# Calculate a similarity score using some formula
# Recommend the items with the highest score
#这个算法实现和我想的不太一样


    def getScore(history,similarities):
        return sum(history * similarities)/sum(similarities)

#行索引为用户,列索引为歌曲名,data.index不等于用户编号
    data_sims = pd.DataFrame(index=data.index,columns=data.columns)

#data_sims的用户列进行初始化
    data_sims.ix[:,:1] = data.ix[:,:1]

    data_germany = data.drop('user',1)
    data_ibs = pd.DataFrame(index=data_germany.columns,columns=data_germany.columns)

    for i in range(0,len(data_sims.index)):
        for j in range(1,len(data_sims.columns)):
            user = data_sims.index[i]
            product = data_sims.columns[j]

#相应歌曲被打分过,忽略
            if data.ix[i][j] == 1:
                data_sims.ix[i][j] = 0
            else:
                #和当前歌曲相似度最高的十首
                product_top_names = data_neighbours.ix[product][1:10]
                #和当前歌曲相似度最高的十首的cos距离
                product_top_sims = data_ibs[product].order(ascending=False)[1:10]
                #用户是否购买相应歌曲
                user_purchases = data_germany.ix[user,product_top_names]

                #通过购买情况和相似度计算用户相应歌曲的分数
                data_sims.ix[i][j] = getScore(user_purchases,product_top_sims)

#
    data_recommend = pd.DataFrame(index=data_sims.index,columns=['user','1','2','3','4','5','6'])
#第0列是user列,下面语句就是初始化该列,注意第0列不是index列.
    data_recommend.ix[:,0] = data_sims.ix[:,0]


    for i in range(0,len(data_sims.index)):
        data_recommend.ix[i,1:] = data_sims.ix[i,:].order(ascending=False).ix[1:7,].index.transpose()

    print data_recommend.ix[:10,:4]

if __name__ == "__main__":
     data = loadData("small.csv")
     data_neighbours = itemBasedCF(data)

     userBasedCF(data,data_neighbours)

    # print data_neighbours.head(6).ix[:6,2:4]


