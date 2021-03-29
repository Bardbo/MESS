# -*- coding: utf-8 -*-
# @Author: Bardbo
# @Date:   2020-11-16 17:37:12
# @Last Modified by:   Bardbo
# @Last Modified time: 2020-11-23 13:13:07
import osmnx as ox
import networkx as nx
import igraph as ig
import geatpy as ea
import pandas as pd
import numpy as np
import multiprocessing as mp
import time
import os


class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        M = 1  # 初始化M（目标维数）
        maxormins = [1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = 6  # 初始化Dim（决策变量维数）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = [0.5] * Dim # 决策变量下界
        ub = [30] * Dim # 决策变量上界
        lbin = [1] * Dim # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
        self.edges_data = pd.read_csv('myCode\multi\edges_data.csv')

    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 得到决策变量矩阵
        x1 = Vars[:, [0]]
        x2 = Vars[:, [1]]
        x3 = Vars[:, [2]]
        x4 = Vars[:, [3]]
        x5 = Vars[:, [4]]
        x6 = Vars[:, [5]]
        nums = Vars.shape[0]
        loss_ls = []
        for i in range(nums):
            self.speed = {'unclassified':x1[i], 'tertiary':x2[i], 'primary':x3[i], 'secondary':x4[i], 'residential':x5[i], 'else':x6[i]}
            edges_data = (self.edges_data['highway'].apply(self.get_speed)).tolist()
            pd.DataFrame(edges_data, columns=['highway']).to_csv('myCode\multi\speed.csv', index=False)
            if os.path.exists('myCode\multi\loss.csv'):
                os.remove('myCode\multi\loss.csv')
            os.system("python D:\大论文\myCode\9.计算目标函数值.py")
            exi = True
            while exi:
                if os.path.exists('myCode\multi\loss.csv'):
                    exi = False
                    loss = pd.read_csv('myCode\multi\loss.csv').iloc[0, 0]
            loss_ls.append(loss)
        pop.ObjV = np.array(loss_ls).reshape((nums, 1))  # 计算目标函数值，赋值给pop种群对象的ObjV属性
        

    def get_speed(self, string:str):
        return self.speed.get(string, self.speed['else'])
    

"""================================实例化问题对象==========================="""
problem = MyProblem()  # 生成问题对象
"""==================================种群设置=============================="""
Encoding = 'RI'  # 编码方式
NIND = 20  # 种群规模
Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)  # 创建区域描述器
population = ea.Population(Encoding, Field, NIND)  # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
"""================================算法参数设置============================="""
myAlgorithm = ea.soea_DE_rand_1_bin_templet(problem, population)  # 实例化一个算法模板对象
myAlgorithm.MAXGEN = 100  # 最大进化代数
myAlgorithm.mutOper.F = 0.5  # 差分进化中的参数F
myAlgorithm.recOper.XOVR = 0.7  # 重组概率
myAlgorithm.logTras = 1  # 设置每隔多少代记录日志，若设置成0则表示不记录日志
myAlgorithm.verbose = True  # 设置是否打印输出日志信息
myAlgorithm.drawing = 1  # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画；3：绘制决策空间过程动画）
"""===========================调用算法模板进行种群进化========================"""
[BestIndi, population] = myAlgorithm.run()  # 执行算法模板，得到最优个体以及最后一代种群
BestIndi.save()  # 把最优个体的信息保存到文件中
"""==================================输出结果=============================="""
print('评价次数：%s' % myAlgorithm.evalsNum)
print('时间已过 %s 秒' % myAlgorithm.passTime)
if BestIndi.sizes != 0:
    print('最优的目标函数值为：%s' % BestIndi.ObjV[0][0])
    print('最优的控制变量值为：')
    for i in range(BestIndi.Phen.shape[1]):
        print(BestIndi.Phen[0, i])
else:
    print('没找到可行解。')