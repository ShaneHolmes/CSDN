<table>
    <thead>
        <tr>
            <th colspan=3>论文杂记</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td bgcolor="#EEEEEE"><a href="" target="_blank">上一篇</a></td>
            <td bgcolor="#CFCFFC"><a href="https://blog.csdn.net/qq_33208851/article/details/100178773" target="_blank">主目录</a> </td>
            <td bgcolor="#EEEEEE"><a href="" target="_blank">下一篇</a></td>
        </tr>
    </tbody>
</table>

***
@[TOC](文章结构)  
   【**前言**】
   受图像采集过程中光照变化、形状和噪声等因素影响，**基于区域或轮廓**的方法往往会出现若干错误。基于区域的方法，一般是利用**颜色和纹理特征**来表示区域，该方法可以抗区域大小、平移与旋转等变化，但由于这些特征**不包括图像像素间空间位置特性**，因此不能较好地进行图像识别。**图像的复杂网络特征具有较强的稳定与抗噪能力**，因此，提出一种图像的有向复杂网络表示模型，利用**K近邻（KNN）**确定有向复杂网络的演化序列，并利用**复杂网络的度平均与熵**等参数完成图像的轮廓识别。
  
   **要点概述：**
   1. 基于KNN使用有向复杂网络来识别图像轮廓识别
   2. KNN算法基础
   3. 查全率和查准率来度量检索结果的好坏

   关键词：**复杂网络；图像识别；图像轮廓；K最近邻；熵**
论文《基于KNN有向复杂网络的图像轮廓识别》[下载地址](http://cdmd.cnki.com.cn/Article/CDMD-10270-1016133523.htm)
或者联系博主获取，邮箱：shaneholmes@qq.com

**注意：**
1.本文涉及到的**复杂网络**的概念，基于复杂网络的**图像形状轮廓识别**方法，基于有向复杂网络模型的**图像特征提取**方法等基础知识在上海师范大学一篇硕士毕业论文里有详细介绍，建议先看这篇论文再学习2.2节提取特征向量部分。  <font color="red">【推荐阅读】</font>。
2.论文《Image Modeling and Feature Extraction Method Based on Complex Netword》[下载链接](https://download.csdn.net/download/qq_33208851/11859389)，或联系博主获取，邮箱 shaneholmes@qq.com
***
## 1 预备知识
### 1.1查全率和查准率
   <font color="red">查全率</font>是指，检索出的相关记录与全部相关记录之间的比值，<font color="red">查准率</font>是指检索出的相关记录与检索出的全部记录的比值，从查全-查准率曲线的分布我们可以判断图像检索算法的性能。当查全-查准率曲线与两条坐标轴之间所围成的面积越大，则检索性能越好。
- 【例子】假设从图库中检索某个类别的一张图片，该类别有1000张图片，检索结果是检索出800张图片，其中700张是该类的图片，而其他100张不属于该类别。那么**查全率=700/1000**，**查准率=700/800**

 ### 1.2 KNN
详情见： [机器学习之最邻近规则分类KNN](https://blog.csdn.net/qq_33208851/article/details/91366297)
### 1.3  水平差分和垂直差分
在图像中<font color="red">差分算子</font>的概念：
```python
#水平差分算子
#作用：检测水平方向上的灰度值变化
[[-1  0  1]
 [-1  0  1]
 [-1  0  1]](3X3)
```

```python
#垂直差分算子
#作用：检测垂直方向上的灰度值变化
[[-1 -1  -1]
 [0   0   0]
 [1   1   1]](3X3)
```
- 【作用原理】
滤波和卷积是有差异的，详情参见博客
https://blog.csdn.net/hellocsz/article/details/101177489
https://www.cnblogs.com/xiaojianliu/p/9075872.html
以水平差分算子为例：
[[-1  0  1]
 [-1  0  1]
 [-1  0  1]] (3X3)
 相当于一个3X3的滤波器，对以下image进行滤波
 [[1 ...1 1 1 10 ...10]
 [1 ...1 1 1 10 ...10]
 [...   ]
 [1 ...1 1 1 10 ...10]
 [1 ...1 1 1 10 ...10]
 [1 ...1 1 1 10 ...10]] (10X10)
结果为(自己算，大概就是下面这样)：
 [ [...   ]
 [ ... 0 27 27 0...]
 [ ... 0 27 27 0...]
 [ ... 0 27 27 0...]
  [...   ]] (10X10)
  在image中，水平方向上存在灰度值的差异（从1~10的过渡带），然后使用水平差分算子（3X3的滤波器）去进行滤波，会得到如上的结果，这个滤波器能够反映出水平方向上灰度值的变化。

### 1.4 <font color="red">Harris角点检测</font>
[B站视频讲解](https://www.bilibili.com/video/av45011346?from=search&seid=5019923561762643832)【建议观看】
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191012221803812.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMzMjA4ODUx,size_16,color_FFFFFF,t_70#pic_center)
**【算法思想】**
为了判断图像的角点，可以利用窗口滑动的思想，让以该点为中心的窗口在附近滑动，**当滑动窗口在所有方向移动时，窗口内的像素灰度出现了较大的变化，就可能是角点。**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191012221851423.png#pic_center)
**【公式的理解】**
- w(x,y)是一种加权函数，几乎所有的应用都把它设为高斯函数
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191012222654820.png#pic_center)
- u,v：u，v分别是x水平方向移动的距离大小，y垂直方向移动的距离大小，可为负数
- I(x,y)是在(x,y)点处的灰度值大小；I(x+u,y+v)是经过水平方向移动u，竖直方向上移动y后在(x+u,y+v)点处的灰度值大小
- I(x+u,y+v)-I(x,y)是移动后对应位置的灰度值差
- 【I(x+u,y+v)-I(x,y)】^2是为了出现负数
- 每个对应位置的灰度值差乘以加权值
- ∑对窗口内所有像素点的操作求和，最后得到的E(u,v)是一个数


**【推导公式】**

由已知：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191012225000793.png#pic_center)
展开：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191012224137213.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMzMjA4ODUx,size_16,color_FFFFFF,t_70#pic_center)
所以：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191012224323671.png#pic_center)
M是实对称矩阵，必可正交化：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191012232524427.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMzMjA4ODUx,size_16,color_FFFFFF,t_70#pic_center)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191012232810948.png#pic_center)
- E(u,v)是E关于u，v的二次函数。实际上是一个椭圆
- Ix是函数I(x,y)对x的偏导数，Iy同理（关于[离散多元函数的偏导数](https://www.ixueshu.com/document/04c21b4cdc11a327318947a18e7f9386.html)）
- M是2X2的矩阵
- E(u,v)是一个数=[1,2] [2,2] [2,1]

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191012233153797.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMzMjA4ODUx,size_16,color_FFFFFF,t_70#pic_center)

   <font color="red">用下式表示λ1和λ2同时很大且相近（此时R为大数值正数）：</font>（不唯一）
   ![在这里插入图片描述](https://img-blog.csdnimg.cn/20191012233636131.png#pic_center)
   - det M是M的行列式，trace M是M的迹
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191012233824208.png#pic_center)
- k是常量，取值范围为0.04~0.06

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191013103410738.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMzMjA4ODUx,size_16,color_FFFFFF,t_70#pic_center)

### 1.5 <font color="red">KNN演化模型</font>
动态演化是复杂网络的一个非常重要的特征， 但是以往的演化模型， 如阈值演化、 一般 MST 演化等都是基于无向网络模型的， 即演化后生成的子网络都是无向子网络。 众所周知， 有向图往往包含更加丰富的结构信息。 一种新的基于 KNN 方法的演化模型， 使得演化后生成的子网络为**有向子网络**。 对于初始的网络G0 ， 具体演化过程如下：

> ![在这里插入图片描述](https://img-blog.csdnimg.cn/20191013165109913.png#pic_center)
> - 其中 k=1， 2， ...， |v|-1， **KNN(i)定义为节点 i 的 k 近邻**。 |v|是点的个数，等于n。**K从1取值到|v|-1是因为对于某个点i，还有其他|v|-1个点要与它建立有向边的关系**（如何建立？？？如下）
> - 如果节点 i 是节点 j 的近邻， 则从节点 i 到节点 j 之间有一条有向边相连。 （从谁到谁？？？边的权值？？？）
> - 这样对于不同的 k 值，就会得到一系列演化后的有向子网络。 图 4-1 显示了图像及其有向复杂网络动态演化的过程， 其中左边的两幅子图显示的是初始图像以及对应的特征点， 右边的六幅子图分别对应K= 3， 5， 7, 9， 11， 13 时的有向子网络。
> 
> ![在这里插入图片描述](https://img-blog.csdnimg.cn/20191013165219601.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMzMjA4ODUx,size_16,color_FFFFFF,t_70#pic_center)


## 2 图像描述与特征提取
### 2.1 算法流程
**算法步骤**：

>1.利用Harris角点检测法提取图像中的关键点，并构建初始网络模型； 
> 2.利用KNN实现复杂网络的动态演化，从而得到有向子网络序列；
> 3.利用有向子网络中的度平均、熵等特征构成特征向量； 
> 4.利用特征向量完成图像轮廓的识别。

![在这里插入图片描述](https://img-blog.csdnimg.cn/2019101317385087.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMzMjA4ODUx,size_16,color_FFFFFF,t_70#pic_center)
### 2.2 图像的复杂网络表示（Harris角点检测，构建网络模型）
 <font color="red">**可以使用图G(V,E)表示图像 I**</font>。其中，V和E分别表示图中顶点的集合以及边的集合（应该是一个**无向图**）
 
 **【如何使用图G(V,E)表示图像 I】**
 

> 1.使用Harris提出图像中的角点，即n个关键点
> 2.这 n 个关键点作为网络的顶点，来建立网络模型Gn=(Vn,En)
> 3.计算任意两个节点i 与j 间的连接边的权值，用两者的欧氏距离d(i,j)表示，即：wij= d(i,j)
> 	4.网络模型Gn可以用一个n×n的权值矩阵M来表示（实对称矩阵）
> 5.将权值归一化：
> ![在这里插入图片描述](https://img-blog.csdnimg.cn/20191013110331610.png#pic_center)
> Wij=d（i，j）>=0,每个值除以最大值就可以将矩阵M所有的元素值变成大于零小于一的数，即归一化
> 

【问题思考】
- 使用Harris提出图像中的角点，即n个关键点作为图G的角，这n个关键点是什么东西？如何表示？
 <font color="red">角点就是图像I中的某些像素点，在原图像I上找到n个角点，就是在在原图像上找到n个不同的像素点</font>
 -  图G的边如何表示？
~~整幅图像其实就是在二维坐标系下的离散的 I(x,y)，相邻的两个像素的距离就是1。确定了每个像素点的x，y位置坐标，就可以计算欧氏距离~~ （好像不是这样的。。。？？？）
顶点i和j之间的欧式距离就是两者之间连接边的权值，再KNN演化中会根据边权值的大小来生成有向子网络。边权值大小的表示

### 2.3 提取特征向量（KNN演化，特征提取）
经过2.1，我们已经将图像I转化为图G，并用矩阵M来表示G，其中矩阵的元素值是角点i，j之间的欧氏距离的归一化的值
- 【注】特征向量对于图像处理是及其重要的，图像特征的正确性、完整性会影响着算法的好坏


**特征提取步骤：**

> 1.用KNN对上述网络（图G）进行动态演化，从而得到有向子网络序列 
> 2.并将其串联构成一特征向量，即
> ![在这里插入图片描述](https://img-blog.csdnimg.cn/20191013173402668.png#pic_center)
> ![在这里插入图片描述](https://img-blog.csdnimg.cn/20191013173441939.png#pic_center)
> 其中， 分别表示第i个有向子网络的度均值、熵值和能量。这里，度均值是将有向子网络中的各节点的度相加再计算其平均值。熵和能量的计算公式如下：
> ![在这里插入图片描述](https://img-blog.csdnimg.cn/2019101317352854.png#pic_center)
> ![在这里插入图片描述](https://img-blog.csdnimg.cn/20191013173540513.png#pic_center)
> 

## 3 实验结果
**实验数据：** Columbia Object Image Library-100(COIL-100)图像库。
该图像库共包括 100 种类别的物体，每个类别中各包含72幅从不同视角拍摄的图像，总共有7200幅。本文共选取8种物体，每种物体包括15 幅图像，总共组成 120 幅图像作检索，并与基于EWT(Edges Weights Threshold)[3]和基于 GED(Graph Edit Distance)[4]的图像描述方法进行对比。实验中，每幅图像共提取45个关键点。本文利用查全率与查准率来说明检索结果[5]，当查全—查准率曲线与坐标轴围成的面积越大，则说明检索性能越好。
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019101317443943.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMzMjA4ODUx,size_16,color_FFFFFF,t_70#pic_center)
**结果**：
本文提出了基于KNN的有向复杂网络模型来进行图像形状的识别，并在COIL-100图像库上选取8种物体，共120幅图像进行了检索实验。实验结果，相比于基于EWT和基于GED方法，本文提出的方法查全率与查准率均高于其他方法。


