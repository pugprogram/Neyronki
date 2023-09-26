import numpy as np
import cv2
import matplotlib.pyplot as plt
import math


def DownloadImage(name):
    color_image = cv2.imread(name)
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    n_image = np.around(np.divide(gray_image, 255.0), decimals=1)
    for i in range (0,len(n_image)):
        for j in range (0,len(n_image[0])):
            if (n_image[i][j]==1):
                n_image[i][j]=0
            else:
                n_image[i][j]=1
    w=[0 for __ in range(20)]
    k=0
    for i in range (len(n_image)):
        for j in range (len(n_image[0])):
            w[k]=n_image[i][j]
            k+=1
    return w


def funcActivate(name,value,w):
    sum=0
    for i in range (len(value)):
        sum+=w[i]*value[i]
    if name=="Linear":
        return round(sum)
    elif name=="Sigmoid":
        return 1/(1 + math.exp(-sum))
    elif name == "Tanh":
        return np.tanh(sum)
    else:
        return max(0,sum)


def Training(nameImage,w,alfa,correctValue,nameFunc,errorarray):
    InputArray=DownloadImage(nameImage)
    output=funcActivate(nameFunc,InputArray,w)
    error=(correctValue-output)**2
    errorarray.append(error)
    for i in range (len(w)):
        w[i]+=alfa*error*InputArray[i]
        #print(w[i])
    return w,errorarray



def Learning(NumOfEpoch):
    alfa=0.1
    wLinear=[0 for _ in range (20)]
    wTanh=[0 for _ in range (20)]
    WSigmoid=[0 for _ in range (20)]
    WRelu=[0 for _ in range (20)]


    errorLinear=[]
    errorTanh=[]
    errorSigmod=[]
    errorRelu=[]

    for _ in range (NumOfEpoch):
        for j in range (1,2):
            nameImage=str(j)+".png"
            wLinear,errorLinear=Training(nameImage,wLinear,alfa,j,"Linear",errorLinear)
            WSigmoid,errorSigmod=Training(nameImage,WSigmoid,alfa,j,"Sigmoid",errorSigmod)
            WRelu,errorRelu=Training(nameImage,WRelu,alfa,j,"Relu",errorRelu)
            wTanh,errorTanh=Training(nameImage,wTanh,alfa,j,"Tanh",errorTanh)
    return wLinear,errorLinear,WSigmoid,errorSigmod,WRelu,errorRelu,wTanh,errorTanh
    

def Check(nameImage,realValue,LearnValue,w,nameFunc):
    InputArray=DownloadImage(nameImage)
    NewValue=funcActivate(nameFunc,InputArray,w)
    print("Func Name:",nameFunc," Real Value:",realValue," Value:",NewValue," Learn Value:",LearnValue)


wLinear,errorLinear,WSigmoid,errorSigmod,WRelu,errorRelu,wTanh,errorTanh=Learning(1000)
Check("1.png",1,1,wLinear,"Linear")
Check("0.png",0,1,wLinear,"Linear")
Check("2.png",2,1,wLinear,"Linear")
Check("1.png",1,1,WSigmoid,"Sigmoid")
Check("0.png",0,1,WSigmoid,"Sigmoid")
Check("2.png",2,1,WRelu,"Sigmoid")
Check("1.png",1,1,WRelu,"Relu")
Check("0.png",0,1,WRelu,"Relu")
Check("2.png",2,1,WRelu,"Relu")
Check("1.png",1,1,wTanh,"Tanh")
Check("0.png",0,1,wTanh,"Tanh")
Check("2.png",2,1,wTanh,"Tanh")

x=[i for i in range(1000)]
plt.plot(x,errorLinear,label="Linear")
plt.plot(x,errorSigmod,label="Sigmoida")
plt.plot(x,errorTanh,label="Tanh")
plt.plot(x,errorRelu,label="Relu")
plt.grid(True)
plt.legend(loc='best')
plt.savefig("Graph.png")
plt.show()
#print(wLinear)

