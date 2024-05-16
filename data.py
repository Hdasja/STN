import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from gurobipy import Model, GRB, quicksum

#设置全局字体
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['font.serif'] = ['Arial']
matplotlib.rcParams['font.family']='sans-serif'  # 默认使用无衬线字体
# 解决负号'-'显示为方块的问题
matplotlib.rcParams['axes.unicode_minus'] = False

class STN(object):  #定义STN类
    def __init__(self):
        # Data
        # ----
        self.tasks = ['R.milling', 'Calcining', 'C.milling']
        self.states = ['Feed', 'Row', 'Clinker', 'Cement']
        self.horizon = 24 # 时间步数

        # Aliases
        self.J = self.tasks.__len__()   #任务数
        self.S = self.states.__len__()  #状态数
        self.T = self.horizon        #时间步数
        self.infeasible = []

        # Recipes and timing
        # fractions of input state s (column) required to execute task j (row),输入原料矩阵
        self.rho_in = np.array([[1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 1, 0]])   #输入原料矩阵

        # fractions of output state s (column) produced by task j (row)，输出产物矩阵
        self.rho_out = np.array([[0, 0, 0],
                                 [1, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 1]])
        # time (in # of steps) required to produce output state s (column) from task j (row)，状态前一步任务到该状态的小时数
        self.P = np.array([[0, 0, 0],
                           [2, 0, 0],
                           [0, 4, 0],
                           [0, 0, 1]])
        # total execution time of task j (row-wise max of P)，任务j执行的总时间，（P 的行最大值）
        self.P_j = np.amax(self.P, axis=1)  # axis=1表示按行求最大值//这个约束还没有添加

        # storage capacity for state s, 状态 s 的存储容量(最大/最小)
        # np.inf或np.infty是NumPy中表示正无穷大的特殊常量。这个值可以用于表示在数值计算中的正无穷大。储存容量上下限
        '''可能出现的参数设置问题'''
        self.C_max = np.array([np.infty, 2000, 5000, 10000])   #状态最大容量
        '''-----------------------------------------------------'''
        self.C_min = np.array([0, 0, 0, 0])  #状态最小容量

        # 分时电价与光伏上网电价数据（元/kWh）
        self.TOU = np.array([0.2749, 0.2749, 0.2749, 0.2749, 0.2749, 0.2749,
                             0.2749, 0.5499, 0.5499, 0.8248, 0.8248, 0.8248,
                             0.5499, 0.5499, 0.5499, 0.5499, 0.5499, 0.8248,
                             0.8248, 0.8248, 0.8248, 0.8248, 0.5499, 0.2749])
        self.price = np.array([0.4150, 0.4150, 0.4150, 0.4150, 0.4150, 0.4150,
                               0.4150, 0.4150, 0.4150, 0.4150, 0.4150, 0.4150,
                               0.4150, 0.4150, 0.4150, 0.4150, 0.4150, 0.4150,
                               0.4150, 0.4150, 0.4150, 0.4150, 0.4150, 0.4150])
        # 光伏出力数据(kW)
        '''这里必要时也需要修改'''
        self.PV = np.array([0, 0, 0, 0, 0, 0,
                            0, 0, 0, 962, 5400, 2863,
                            3028, 10060, 1998, 1721, 3000, 914,
                            207, 0, 0, 0, 0, 0])

        # Optimization problem structure (cvx.Problem type)
        self.model = Model("my_model")

        # Optimization Variables
        # ----------------------
        # Feed, Row, Clinker, Cement的储存量
        self.P_buy = {}
        self.P_sell = {}
        self.y_Feed = {}
        self.y_Row = {}
        self.y_Clinker = {}
        self.y_Cement = {}
        self.rhoR = {}
        self.productR = {}
        self.powerR = {}
        self.rhoCa = {}
        self.productCa = {}
        self.powerCa = {}
        self.rhoC = {}
        self.productC = {}
        self.powerC = {}
        self.Calcining_kiln_sets_0 = {}
        self.Calcining_kiln_sets_1 = {}
        self.R_mill_sets_0 = {}
        self.R_mill_sets_1 = {}
        self.R_mill_sets_2 = {}
        self.C_mill_sets_0 = {}
        self.C_mill_sets_1 = {}
        self.C_mill_sets_2 = {}


        # Optimization Results
        for t in range(self.T):
            self.P_buy[t] = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0,  name=f"P_buy_{t}")
            self.P_sell[t] = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0,  name=f"P_sell_{t}")
            self.y_Feed[t] = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"y_Feed_{t}")
            self.y_Row[t] = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=2000, name=f"y_Row_{t}")
            self.y_Clinker[t] = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=5000, name=f"y_Clinker_{t}")
            self.y_Cement[t] = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=10000, name=f"y_Cement_{t}")
            self.Calcining_kiln_sets_0[t] = self.model.addVar(vtype=GRB.BINARY, name=f"Calcining_kiln_sets_0_{t}")
            self.Calcining_kiln_sets_1[t] = self.model.addVar(vtype=GRB.BINARY, name=f"Calcining_kiln_sets_1_{t}")
            self.R_mill_sets_0[t] = self.model.addVar(vtype=GRB.BINARY, name=f"R_mill_sets_0_{t}")
            self.R_mill_sets_1[t] = self.model.addVar(vtype=GRB.BINARY, name=f"R_mill_sets_1_{t}")
            self.R_mill_sets_2[t] = self.model.addVar(vtype=GRB.BINARY, name=f"R_mill_sets_2_{t}")
            self.C_mill_sets_0[t] = self.model.addVar(vtype=GRB.BINARY, name=f"C_mill_sets_0_{t}")
            self.C_mill_sets_1[t] = self.model.addVar(vtype=GRB.BINARY, name=f"C_mill_sets_1_{t}")
            self.C_mill_sets_2[t] = self.model.addVar(vtype=GRB.BINARY, name=f"C_mill_sets_2_{t}")
            self.rhoR[t] = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"rhoR_{t}")
            self.productR[t] = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"productR_{t}")
            self.powerR[t] = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"powerR_{t}")
            self.rhoCa[t] = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"rhoCa_{t}")
            self.productCa[t] = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"productCa_{t}")
            self.powerCa[t] = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"powerCa_{t}")
            self.rhoC[t] = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"rhoC_{t}")
            self.productC[t] = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"productC_{t}")
            self.powerC[t] = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"powerC_{t}")




    def constraint_var_mill(self,T,model,productR,powerR,rhoR,R_mill_sets_0,R_mill_sets_1,R_mill_sets_2,P_j,TAB):
        for t in range(T):
            # 添加约束，磨机的选择可调节
            model.addConstr(R_mill_sets_0[t] + R_mill_sets_1[t] + R_mill_sets_2[t] == 1)

            model.addConstr(rhoR[t] == TAB[0] * R_mill_sets_0[t] + TAB[3] * R_mill_sets_1[t] + TAB[6] * R_mill_sets_2[t])
            model.addConstr(productR[t] == TAB[1] * R_mill_sets_0[t] + TAB[4] * R_mill_sets_1[t] + TAB[7] * R_mill_sets_2[t])
            model.addConstr(powerR[t] == TAB[2] * R_mill_sets_0[t] + TAB[5] * R_mill_sets_1[t] + TAB[8] * R_mill_sets_2[t])
            # 添加约束
            # t为奇数时，self.rhoR[t]等于self.rhoR[t-1]
            if t % P_j[1] == 1:
                model.addConstr(rhoR[t] == rhoR[t - 1])
                model.addConstr(productR[t] == productR[t - 1])
                model.addConstr(powerR[t] == powerR[t - 1])


    def constraint_var_Calcining(self,T,model,productCa,powerCa,rhoCa,Calcining_kiln_sets_1,TAB):
        for t in range(T):
            # 添加约束
            model.addConstr(Calcining_kiln_sets_1[t] == 1)

            model.addConstr(rhoCa[t] == TAB[9] * Calcining_kiln_sets_1[t])#
            model.addConstr(productCa[t] == TAB[10] * Calcining_kiln_sets_1[t])
            model.addConstr(powerCa[t] == TAB[11] * Calcining_kiln_sets_1[t])


    def constraint_var_Cmill(self,T,model,productC,powerC,rhoC,C_mill_sets_0,C_mill_sets_1,C_mill_sets_2,TAB):
        for t in range(T):
            # 添加约束，磨机的选择可调节
            model.addConstr(C_mill_sets_0[t] + C_mill_sets_1[t] + C_mill_sets_2[t] == 1)

            model.addConstr(rhoC[t] == TAB[12] * C_mill_sets_0[t] + TAB[15] * C_mill_sets_1[t] + TAB[18] * C_mill_sets_2[t])
            model.addConstr(productC[t] == TAB[13] * C_mill_sets_0[t] + TAB[16] * C_mill_sets_1[t] + TAB[19] * C_mill_sets_2[t])
            model.addConstr(powerC[t] == TAB[14] * C_mill_sets_0[t] + TAB[17] * C_mill_sets_1[t] + TAB[20] * C_mill_sets_2[t])


    def state_constraint(self,T,model,y_Feed,y_Row,y_Clinker,y_Cement,rhoR,productR,rhoCa,productCa,rhoC,productC,Initial):  # 物料平衡约束和储存约束
        # 1) 物料初始条件
        model.addConstr(y_Feed[0] == Initial[0])
        model.addConstr(y_Row[0] == Initial[1])
        model.addConstr(y_Clinker[0] == Initial[2])
        model.addConstr(y_Cement[0] == Initial[3])

        for t in range(1, T):
            model.addConstr(y_Feed[t] == y_Feed[t - 1] - rhoR[t])
            model.addConstr(y_Row[t] == y_Row[t - 1] + productR[t] - rhoCa[t])
            model.addConstr(y_Clinker[t] == y_Clinker[t - 1] + productCa[t] - rhoC[t])
            model.addConstr(y_Cement[t] == y_Cement[t - 1] + productC[t])

        model.addConstr(y_Cement[T - 1] >= 4000)



    def power_constraint(self,T,model,P_buy,PV,P_sell,powerR,powerCa,powerC):  # 功率约束
        for t in range(T):
            model.addConstr(
                P_buy[t] + PV[t] == P_sell[t] + powerR[t] + powerCa[t] + powerC[t])
            #self.model.addGenConstrMax(self.P_buy[t], [self.P_buy[t], self.P_sell[t]], 0)
            # 这里的约束应该还需要修改
            model.addConstr(P_buy[t]*P_sell[t] == 0)
            # 添加约束
            #self.model.addConstr(self.P_buy[t] >= self.P_sell[t] + 1)


    def construct_objective(self,T,model,TOU,P_buy,price,P_sell):

        model.setObjective(
            sum(TOU[t] * P_buy[t] for t in range(T)) - sum(price[t] * P_sell[t] for t in range(T)),
            GRB.MINIMIZE)

    def solve(self,T,model,TOU,price,P_buy,P_sell,y_Feed,y_Row,y_Clinker,y_Cement):
        print('Constructing nominal model...')
        model.setObjective(
            sum(TOU[t] * P_buy[t] for t in range(T)) - sum(price[t] * P_sell[t] for t in range(T)),
            GRB.MINIMIZE)
        print('Solving...')
        model.optimize()
        print("About to print variable values...")

        # 模型不可求解时，输出模型的IIS，可打印出模型的不可行约束
        if self.model.status != 2:
            if self.model.status == GRB.INFEASIBLE:
                print('The model is infeasible; writing .lp file...')
                lis = self.model.computeIIS
                self.infeasible = [constraint.name for constraint in lis]
                self.model.write("model_infeasible.lp")
            return self.infeasible

        else:

            for t in range(T):
                print(f"P_buy at time {t}: ", P_buy[t])
            for t in range(T):
                print(f"P_sell at time {t}: ", P_sell[t])
            for t in range(T):
                print(f"y_Feed at time {t}: ", y_Feed[t])
            for t in range(T):
                print(f"y_Row at time {t}: ", y_Row[t])
            for t in range(T):
                print(f"y_Clinker at time {t}: ", y_Clinker[t])
            for t in range(T):
                print(f"y_Cement at time {t}: ", y_Cement[t])

            return model.objVal

    def export_results(self,T,P_buy,P_sell,y_Feed,y_Row,y_Clinker,y_Cement,rhoR,productR,rhoCa,productCa,rhoC,productC,powerR,powerC,powerCa):
        results = {}
        for t in range(T):
            results.setdefault(f"P_buy", []).append(P_buy[t].X)
            results.setdefault(f"P_sell", []).append(P_sell[t].X)
            results.setdefault(f"y_Feed", []).append(y_Feed[t].X)
            results.setdefault(f"y_Row", []).append(y_Row[t].X)
            results.setdefault(f"y_Clinker", []).append(y_Clinker[t].X)
            results.setdefault(f"y_Cement", []).append(y_Cement[t].X)
            results.setdefault(f"rhoR", []).append(rhoR[t].X)
            results.setdefault(f"productR", []).append(productR[t].X)
            results.setdefault(f"rhoCa", []).append(rhoCa[t].X)
            results.setdefault(f"productCa", []).append(productCa[t].X)
            results.setdefault(f"rhoC", []).append(rhoC[t].X)
            results.setdefault(f"productC", []).append(productC[t].X)
            results.setdefault(f"powerR", []).append(powerR[t].X)
            results.setdefault(f"powerC", []).append(powerC[t].X)
            results.setdefault(f"powerCa", []).append(powerCa[t].X)

        # Convert the dictionary to a DataFrame
        df = pd.DataFrame(results)

        file_path ='optimization_results_Cemment_low.xlsx'

        # Export the DataFrame to an Excel file
        df.to_excel(file_path, index=False)


    def plot_results(self,T,P_buy,P_sell,TOU,price):
        # Create a new figure and a subplot
        fig, ax1 = plt.subplots(dpi=600)

        # Plot P_buy and P_sell
        ax1.bar(range(T), [P_buy[t].X for t in range(T)], label='P_buy', color='deepskyblue')
        ax1.bar(range(T), [-P_sell[t].X for t in range(T)], label='P_sell', color='darkorange')

        # Add a legend
        ax1.legend(loc='upper left')

        # Add labels and title
        ax1.set_xlabel('时间/(h)')
        ax1.set_ylabel('功率/(kW)')

        # Set y-axis ticks for positive and negative values
        ax1.set_yticks(list(np.arange(-1000, 14001, 2000)))  # 设置左边y轴刻度线的最大值为14000
        ax1.set_ylim(-1000, 13000)  # 设置左边y轴的范围

        # Set x-axis ticks
        ax1.set_xticks(range(24))

        # Create a new y-axis that shares the same x-axis
        ax2 = ax1.twinx()

        # Remove the top border
        ax1.spines['top'].set_visible(False)
        ax2.spines['top'].set_visible(False)

        # Plot TOU and price
        ax2.plot(range(T), TOU, label='TOU', color='blue', marker='^')
        ax2.plot(range(T), price, label='PV_Price', color='red', marker='d')

        # Add a legend
        ax2.legend(loc='upper right')

        # Add a label for the new y-axis
        ax2.set_ylabel('电价/(元/kWh)')

        # Set y-axis ticks for positive and negative values
        ax2.set_ylim(0, 1.0)  # 设置右边y轴的范围

        # Set y-axis tick parameters
        ax1.tick_params(axis='y', direction='in')
        ax2.tick_params(axis='y', direction='in')
        # Add y=0 line
        ax1.axhline(0, color='black', linewidth=0.75)
        ax2.axhline(0, color='black', linewidth=0.75)

        # Show the plot


    def plot_y_Feed(self,T,y_Feed):
        plt.figure(figsize=(8, 5), dpi=600)
        plt.bar(range(T), [y_Feed[t].X for t in range(T)], label='原料', color='darkorange')

        # 添加折线图，数据与柱状图一致
        plt.plot(range(T), [y_Feed[t].X for t in range(T)], color='deepskyblue', marker='^')

        plt.xticks(range(0, 24, 1))
        plt.legend(loc='upper right')
        plt.xlabel('时间/(h)')
        plt.ylabel('储存量/(t)')

        ax = plt.gca()  # 获取当前的axes
        ax.spines['right'].set_visible(False)  # 移除右侧的边框
        ax.spines['top'].set_visible(False)  # 移除上侧的边框
        ax.tick_params(axis='y', direction='in')
        ax.set_ylim(0, 9000)  # 设置左边y轴的范围


    def plot_y_Row(self,T,y_Row):
        plt.figure(figsize=(8, 5), dpi=600)
        plt.bar(range(T), [y_Row[t].X for t in range(T)], label='生料', color='skyblue')
        # 添加折线图，数据与柱状图一致
        plt.plot(range(T), [y_Row[t].X for t in range(T)], color='coral', marker='^')
        plt.xticks(range(0, 24, 1))
        plt.legend(loc='upper right')
        plt.xlabel('时间/(h)')
        plt.ylabel('储存量/(t)')
        ax = plt.gca()  # 获取当前的axes
        ax.spines['right'].set_visible(False)  # 移除右侧的边框
        ax.spines['top'].set_visible(False)  # 移除上侧的边框
        ax.tick_params(axis='y', direction='in')
        ax.set_ylim(0, 600)  # 设置左边y轴的范围


    def plot_y_Clinker(self,T,y_Clinker):
        plt.figure(figsize=(8, 5), dpi=600)
        plt.bar(range(T), [y_Clinker[t].X for t in range(T)], label='熟料', color='gold')
        plt.plot(range(T), [y_Clinker[t].X for t in range(T)], color='cyan', marker='^')
        plt.xticks(range(0, 24, 1))
        plt.legend(loc='upper left')
        plt.xlabel('时间/(h)')
        plt.ylabel('储存量/(t)')
        ax = plt.gca()  # 获取当前的axes
        ax.spines['right'].set_visible(False)  # 移除右侧的边框
        ax.spines['top'].set_visible(False)  # 移除上侧的边框
        ax.tick_params(axis='y', direction='in')
        ax.set_ylim(0, 1600)  # 设置左边y轴的范围


    def plot_y_Cement(self,T,y_Cement):
        plt.figure(figsize=(8, 5), dpi=600)
        plt.bar(range(T), [y_Cement[t].X for t in range(T)], label='水泥', color='lightgreen')
        plt.plot(range(T), [y_Cement[t].X for t in range(T)], color='darkorange', marker='^')
        plt.xticks(range(0, 24, 1))
        plt.legend(loc='upper left')
        plt.xlabel('时间/(h)')
        plt.ylabel('储存量/(t)')
        ax = plt.gca()  # 获取当前的axes
        ax.spines['right'].set_visible(False)  # 移除右侧的边框
        ax.spines['top'].set_visible(False)  # 移除上侧的边框
        ax.tick_params(axis='y', direction='in')
        ax.set_ylim(0, 5000)  # 设置左边y轴的范围

    def plot_power_results(self,T,powerR,powerC,powerCa,TOU):
        # Create a new figure and a subplot
        fig, ax1 = plt.subplots(dpi=600)

        # 绘制第一个功率的柱状图
        plt.bar(range(T), [powerR[t].X for t in range(T)], label='生料磨', color='darkorange')

        # 在第一个功率的柱状图之上绘制第二个功率的柱状图
        plt.bar(range(T), [powerC[t].X for t in range(T)], label='立窑、回转窑', color='deepskyblue',
                bottom=[powerR[t].X for t in range(T)])

        # 在第一个和第二个功率的柱状图之上绘制第三个功率的柱状图
        plt.bar(range(T), [powerCa[t].X for t in range(T)], label='水泥磨', color='lightgreen',
                bottom=[powerR[t].X + powerC[t].X for t in range(T)])

        # 添加图例，并设置图例大小
        plt.legend(prop={'size': 8})

        # 计算总功率
        total_power = [powerR[t].X + powerC[t].X + powerCa[t].X for t in range(T)]

        # 绘制总功率
        ax1.plot(range(T), total_power, label='总功率', color='dodgerblue', marker='s')

        # Add a legend，并设置图例大小
        ax1.legend(loc='best', prop={'size': 6})

        # Add labels and title
        ax1.set_xlabel('时间/(h)')
        ax1.set_ylabel('功率/(kW)')

        # Set y-axis ticks for positive and negative values
        ax1.set_ylim(0, 14000)  # 设置左边y轴的范围

        # Set x-axis ticks
        ax1.set_xticks(range(24))

        # Create a new y-axis that shares the same x-axis
        ax2 = ax1.twinx()

        # Plot TOU on the right y-axis using step function
        ax2.step(range(T), TOU, label='TOU', color='tomato', where='post', marker='^')

        # Add a legend，并设置图例大小
        ax2.legend(loc='upper left', prop={'size': 6})

        # Add a label for the new y-axis
        ax2.set_ylabel('电价/(元/kWh)')

        # Set y-axis ticks for positive and negative values
        ax2.set_ylim(0, 1.0)  # 设置右边y轴的范围

        # Remove the top border
        ax1.spines['top'].set_visible(False)
        ax2.spines['top'].set_visible(False)

        # Set y-axis tick parameters
        ax1.tick_params(axis='y', direction='in')
        ax2.tick_params(axis='y', direction='in')

    # Show the plot
