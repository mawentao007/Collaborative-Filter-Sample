#coding:utf-8
__author__ = 'marvin'

from math import sqrt
from PIL import Image,ImageDraw
import random

class bicluster:

    def __init__(self,vec,left=None,right=None,distance=0.0,id=None):
        self.left=left
        self.right=right
        self.vec=vec
        self.id=id
        self.distance=distance

def pearson(v1,v2):
    # Simple sums
    sum1=sum(v1)
    sum2=sum(v2)
    # Sums of the squares
    sum1Sq=sum([pow(v,2) for v in v1])
    sum2Sq=sum([pow(v,2) for v in v2])
    # Sum of the products
    pSum=sum([v1[i]*v2[i] for i in range(len(v1))])
    # Calculate r (Pearson score)
    num=pSum-(sum1*sum2/len(v1))
    den=sqrt((sum1Sq-pow(sum1,2)/len(v1))*(sum2Sq-pow(sum2,2)/len(v1)))
    if den==0: return 0
    return 1.0-num/den


def readfile(filename):
    lines=[line for line in file(filename)]
    # First line is the column titles
    colnames=lines[0].strip( ).split('\t')[1:]
    rownames=[]
    data=[]
    for line in lines[1:]:
        p=line.strip( ).split('\t')
        # First column in each row is the rowname
        rownames.append(p[0])
        # The data for this row is the remainder of the row
        data.append([float(x) for x in p[1:]])
    return rownames,colnames,data


def hcluster(rows,distance=pearson):
    distances = {}
    currentclustid = -1

#开始的时候包含所有的叶子
    clust = [bicluster(rows[i],id = i) for i in range(len(rows))]

    while len(clust) > 1:
        #记录当前最近的两个集群id
        lowestpair=(0,1)
        closest = distance(clust[0].vec,clust[1].vec)

#求所有集群的距离
        for i in range(len(clust)):
            for j in range(i+1,len(clust)):
                if(clust[i].id,clust[j].id) not in distances:
                    distances[(clust[i].id,clust[j].id)]=distance(clust[i].vec,clust[j].vec)

                d = distances[(clust[i].id,clust[j].id)]

                if d < closest:
                    closest = d
                    lowestpair = (i,j)

        # calculate the average of the two clusters
        #距离最近的两个群的向量的平均距离,也就是中间向量,作为新集群的向量.
        mergevec=[
        (clust[lowestpair[0]].vec[i]+clust[lowestpair[1]].vec[i])/2.0
        for i in range(len(clust[0].vec))]

        # create the new cluster
        #创建新的群
        newcluster=bicluster(mergevec,left=clust[lowestpair[0]],
        right=clust[lowestpair[1]],
        distance=closest,id=currentclustid)

        # cluster ids that weren't in the original set are negative
        #集群中删除旧的两个,添加一个新的进去
        currentclustid-=1
        del clust[lowestpair[1]]
        del clust[lowestpair[0]]
        clust.append(newcluster)

    return clust[0]

def printclust(clust,labels=None,n=0):
    # indent to make a hierarchy layout
    for i in range(n): print ' ',
    if clust.id<0:
    # negative id means that this is branch
        print '-'
    else:
    # positive id means that this is an endpoint
        if labels==None: print clust.id
        else: print labels[clust.id]

    # now print the right and left branches
    if clust.left!=None: printclust(clust.left,labels=labels,n=n+1)
    if clust.right!=None: printclust(clust.right,labels=labels,n=n+1)


def getheight(clust):
    # Is this an endpoint? Then the height is just 1
    if clust.left==None and clust.right==None: return 1
    # Otherwise the height is the same of the heights of
    # each branch
    return getheight(clust.left)+getheight(clust.right)

def getdepth(clust):
    # The distance of an endpoint is 0.0
    if clust.left==None and clust.right==None: return 0
    # The distance of a branch is the greater of its two sides
    # plus its own distance
    return max(getdepth(clust.left),getdepth(clust.right))+clust.distance

def drawdendrogram(clust,labels,jpeg='clusters.jpg'):
    # height and width
    h=getheight(clust)*20
    w=1200
    depth=getdepth(clust)
    # width is fixed, so scale distances accordingly
    scaling=float(w-150)/depth
    # Create a new image with a white background
    img=Image.new('RGB',(w,h),(255,255,255))
    draw=ImageDraw.Draw(img)
    draw.line((0,h/2,10,h/2),fill=(255,0,0))
    # Draw the first node
    drawnode(draw,clust,10,(h/2),scaling,labels)
    img.save(jpeg,'JPEG')

def drawnode(draw,clust,x,y,scaling,labels):
    if clust.id<0:
        h1=getheight(clust.left)*20
        h2=getheight(clust.right)*20
        top=y-(h1+h2)/2
        bottom=y+(h1+h2)/2
        # Line length
        ll=clust.distance*scaling
        # Vertical line from this cluster to children
        draw.line((x,top+h1/2,x,bottom-h2/2),fill=(255,0,0))
        # Horizontal line to left item
        draw.line((x,top+h1/2,x+ll,top+h1/2),fill=(255,0,0))
        # Horizontal line to right item
        draw.line((x,bottom-h2/2,x+ll,bottom-h2/2),fill=(255,0,0))
        # Call the function to draw the left and right nodes
        drawnode(draw,clust.left,x+ll,top+h1/2,scaling,labels)
        drawnode(draw,clust.right,x+ll,bottom-h2/2,scaling,labels)
    else:
        # If this is an endpoint, draw the item label
        draw.text((x+5,y-7),labels[clust.id],(0,0,0))

def rotatematrix(data):
    newdata=[]
    for i in range(len(data[0])):
        newrow=[data[j][i] for j in range(len(data))]
        newdata.append(newrow)
    return newdata

def draw2d(data,labels,jpeg='mds2d.jpg'):
    img=Image.new('RGB',(2000,2000),(255,255,255))
    draw=ImageDraw.Draw(img)
    for i in range(len(data)):
        x=(data[i][0]+0.5)*1000
        y=(data[i][1]+0.5)*1000
        draw.text((x,y),labels[i],(0,0,0))
    img.save(jpeg,'JPEG')


def kcluster(rows,distance=pearson,k=4):
    # Determine the minimum and maximum values for each point
    ranges=[(min([row[i] for row in rows]),max([row[i] for row in rows]))
    for i in range(len(rows[0]))]
    # Create k randomly placed centroids
    clusters=[[random.random( )*(ranges[i][1]-ranges[i][0])+ranges[i][0]
    for i in range(len(rows[0]))] for j in range(k)]
    lastmatches=None
    for t in range(100):
        print 'Iteration %d' % t
        bestmatches=[[] for i in range(k)]
        # Find which centroid is the closest for each row
        for j in range(len(rows)):
            row=rows[j]
            bestmatch=0
            for i in range(k):
                d=distance(clusters[i],row)
                if d<distance(clusters[bestmatch],row): bestmatch=i
            bestmatches[bestmatch].append(j)

        # If the results are the same as last time, this is complete
        if bestmatches==lastmatches: break
        lastmatches=bestmatches

        # Move the centroids to the average of their members
        for i in range(k):
            avgs=[0.0]*len(rows[0])
            if len(bestmatches[i])>0:
                for rowid in bestmatches[i]:
                    for m in range(len(rows[rowid])):
                        avgs[m]+=rows[rowid][m]
                for j in range(len(avgs)):
                    avgs[j]/=len(bestmatches[i])
                clusters[i]=avgs
    return bestmatches


def tanimoto_dist(v1, v2):
  c1, c2, shr = 0, 0, 0
  for i in range(len(v1)):
    if v1[i] != 0: c1 += 1
    if v2[i] != 0: c2 += 1
    if v1[i] != 0 and v2[i] != 0: shr += 1
  return 1.0 - float(shr)/(c1 + c2 - shr)


def hypot(v):
  return sqrt(sum([x*x for x in v]))


def euclid_dist(v1, v2):
  return hypot([v[0] - v[1] for v in zip(v1, v2)])

def scaledown(data, distance=pearson, rate=0.01):
  n = len(data)

  realdist = [[distance(data[i], data[j]) for j in range(n)] for i in range(n)]
  outersum = 0.0

  # random start positions
  loc = [[random.random(), random.random()] for i in range(n)]

  lasterror = None
  for m in range(0, 1000):
    # find projected distance
    fakedist = [[euclid_dist(loc[i], loc[j])
      for j in range(n)] for i in range(n)]

    # move points
    grad = [[0.0, 0.0] for i in range(n)]

    totalerror = 0
    for k in range(n):
      for j in range(n):
        if j == k: continue

        # error is percent difference between distances
        errorterm = (fakedist[j][k] - realdist[j][k])/realdist[j][k]

        grad[k][0] += ((loc[k][0] - loc[j][0])/fakedist[j][k]) * errorterm
        grad[k][1] += ((loc[k][1] - loc[j][1])/fakedist[j][k]) * errorterm

        totalerror += abs(errorterm)
    print totalerror

    # if we got worse by moving the points, quit
    if lasterror and lasterror < totalerror: break

    # also break if the improvement is only very small
    if lasterror and lasterror - totalerror < 1e-15: break

    lasterror = totalerror

    # move points by learning rate times gradient
    if k in range(n):
      loc[k][0] -= rate * grad[k][0]
      loc[k][1] -= rate * grad[k][1]

  return loc


if __name__ == '__main__':
    blognames, words, data = readfile('blogdata.txt')
    clust = hcluster(data)
    drawdendrogram(clust, blognames, jpeg='blogclust.jpg')  # 查看哪些单词经常一同出现
    rdata = rotatematrix(data)
    wordClust = hcluster(rdata)
    drawdendrogram(wordClust, labels=words, jpeg='wordclust.jpg')

    coords = scaledown(data)
    draw2d(coords,blognames,jpeg='blogs2d.jpg')
