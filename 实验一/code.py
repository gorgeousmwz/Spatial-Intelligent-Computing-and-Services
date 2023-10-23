from osgeo import gdal
import sys
import numpy as np
from scipy.interpolate import interp2d
from tqdm import tqdm
import os
import random
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import prettytable


class SmallPoint():
    '''小格点类'''
    def __init__(self,lat,lon,value) -> None:
        self.lat=lat # 纬度
        self.lon=lon # 经度
        self.value=value # 甲烷排放强度

class BigPoint():
    '''大格点类'''
    def __init__(self,lat,lon,H) -> None:
        self.lat=lat # 纬度
        self.lon=lon # 经度
        self.H=H # 足迹矩阵

def readXTifFile():
    '''
    读取甲烷通量场tif文件
    returns:
        data: tif文件中的甲烷通量场数据二维矩阵
        origin_lat: 文件左上角纬度
        origin_lon: 文件左上角经度
    '''
    # 打开.tif文件
    dataset = gdal.Open('SICISP2023\\1-INTELLIGENCEALGORITHM\\Global_Fuel_Exploitation_Inventory_v2_2019_Total_Fuel_Exploitation.tif')
    if dataset is None:
        print("!->无法打开文件")
        sys.exit(1)
    # 获取文件的元数据
    metadata = dataset.GetMetadata()
    print('-->原数据:',metadata)
    # 获取文件的宽度和高度
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    print('-->高:',height,' 宽:',width)
    # 获取文件的波段数量
    band_count = dataset.RasterCount
    print('-->波段数:',band_count)
    # 获取经纬度
    geotransform = dataset.GetGeoTransform()# 获取地理转换参数
    # 获取左上角经纬度
    origin_lon = round(geotransform[0],2)
    origin_lat = round(geotransform[3],2)
    print('-->左上角经纬度:',(origin_lon,origin_lat))
    # 获取第一个波段
    band = dataset.GetRasterBand(1)
    # 读取波段数据为二维数组
    data = band.ReadAsArray()
    # 关闭数据集
    dataset = None
    # 剔除data中的负数
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i,j]<0:
                data[i,j]=0
    return data,origin_lat,origin_lon

def readHCsvFile():
    '''
    读取足迹csv文件
    returnd:
        big_points: 一个7x7二维矩阵,每个点存储一个BigPoint
    '''
    big_points= np.empty((7, 7), dtype=BigPoint)
    path='SICISP2023\\1-INTELLIGENCEALGORITHM\\footprints'
    files=os.listdir(path) # 获取文件名
    i,j=6,0
    for file in tqdm(files):
        H = np.genfromtxt(os.path.join(path,file), delimiter=',') # 读取足迹数据
        # 获取经纬度
        str=file.split('_')
        lat=float(str[1]) # 纬度
        lon=float(str[2]) # 经度
        big_point=BigPoint(lat,lon,H) # 创建大格点
        big_points[i,j]=big_point # 添加到大格点矩阵中
        j+=1
        if j==7:
            j=0
            i-=1
    return big_points

def generateSmallPoints(data,origin_lat,origin_lon):
    '''
    生成小格网点
    params:
        data: 数据二维矩阵
        origin_lat: 数据左上角纬度
        origin_lon: 数据左上角经度
    returns:
        small_points: 一个二维矩阵,每个点存储一个SmallPoint
    '''
    small_points=np.empty((data.shape[0],data.shape[1]),dtype=SmallPoint)
    for i in tqdm(range(data.shape[0])):
        for j in range(data.shape[1]):
            value=data[i,j]
            lat=origin_lat-0.01*i
            lon=origin_lon+0.01*j
            small_point=SmallPoint(lat,lon,value)
            small_points[i,j]=small_point
    return small_points

def calY(big_point,point,fitness):
    '''
    计算y值
    params:
        big_point: 观测站所在的大格点
        point: 观测站位置
        fitness: 适应度值字典(每个点位的y值字典)
    return:
        fitness: 修改后的适应度值字典
    '''
    # 计算在每个大格子里面站点的索引
    sub_i=point[0]%10-4
    sub_j=point[1]%10-4
    # 通过大格子中心的经纬度计算站点经纬度
    lat=big_point.lat-sub_i*0.01
    lon=big_point.lon+sub_j*0.01
    # 通过左上角经纬度确定站点在tif中的索引
    i=int(abs(lat-origin_lat)//0.01)
    j=int(abs(lon-origin_lon)//0.01)
    # 遍历H覆盖的区域，点乘求Y值
    for y in range(i-134,i+136):
        for x in range(j-134,j+136):
            Y=big_point.H[y-i+134,x-j+134]*SP[y,x].value # H和甲烷通量对应点值相乘得Y值
            if Y!=0 and fitness.get((y,x),0)<Y: # 如果个点Y=0则忽略;如果这个点之前有值，且小于新的Y，则更新；如果没值则赋值为Y
                fitness[(y,x)]=Y
    return fitness

def getRightBigPoint(point):
    '''
    获取小格点对应的大格点
    params:
        point: 小格点
    returns:
        big_point: 大格点
    '''
    big_point=BP[point[0]//10,point[1]//10]
    return big_point

def individual_decoding(chromosome):
    '''
    染色体解码
    params:
        chromosome: 染色体
    returns:
        individual: 个体
    '''
    individual=[]
    for gene in chromosome:
        individual.append((gene//70,gene%70))
    return individual

def fitness_function(chromosome):
    '''
    适应度函数(目标函数)
    params:
        chromosome: 染色体
    return:
        fitness_value: 适应度值
    '''
    individual=individual_decoding(chromosome) # 解码得到个体
    fitness={} # 适应度字典
    for point in individual:
        big_point=getRightBigPoint(point) # 确定站点所在的大网格
        fitness=calY(big_point,point,fitness) # 计算更新各点位Y值
    fitness_value=sum(list(fitness.values())) # 计算总的Y值
    return fitness_value

def generate_population(solution_num,population_size):
    '''
    生成初始种群
    params:
        solution_num: 染色体长度（求解观测站数目）
        population_siaze: 种群规模
    returns:
        population: 种群
    '''
    population = []
    for _ in range(population_size):
        chromosome = random.sample(range(0, 4900), solution_num) # 生成求解观测站数目个0-4899的整数
        population.append(chromosome)
    return population

def selection(population,fits,population_size):
    '''
    轮盘赌父体选择
    pramas:
        population: 种群
        fits: 适应度值列表
        population_size: 种群规模
    returns:
        selected_population: 被选择的个体组成的种群
    '''
    total_fitness = sum(fits) # 计算染色体适应度值之和
    probabilities = [fitness / total_fitness for fitness in fits] # 计算每个染色体的选择概率
    selected_indices = random.choices(range(population_size), weights=probabilities, k=population_size) # 根据选择概率选择种群规模个父体
    selected_population = [population[i] for i in selected_indices]
    random.shuffle(selected_population)
    return selected_population

def crossover(parent1, parent2):
    '''
    单点交叉,杂交操作
    params:
        parent1: 父体1
        parent2: 父体2
    returns:
        child1: 孩子1
        child2: 孩子2
    '''
    crossover_point = random.randint(0, len(parent1)-1) # 随机生成杂交点位
    # 单点杂交
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def mutation(chromosome,mutation_rate):
    '''
    变异操作
    params:
        chromosome: 染色体
        mutation_rate: 变异概率
    returns:
        mutated_chromosome: 变异后的染色体
    '''
    mutated_chromosome = chromosome.copy()
    for i in range(len(chromosome)):
        if random.random() < mutation_rate: # 保证大致的变异率在mutation_rate附近
            mutated_chromosome[i] = random.randint(0,4899) # 随机生成一个0-4899的整数替代
    return mutated_chromosome

def calLocal_updateGlobal(no_improve_threshold,count_no_improve,last_sum_fitness,global_best_individual,gloabal_best_fitness,population,fits):
    '''
    计算局部参数,更新全局参数
    pramas:
        no_improve_threshold: 没有显著改善的阈值
        count_no_improve: 没有显著提升的代数
        last_sum_fitness: 上一代总适应度值
        global_best_individual: 全局最优个体
        gloabal_best_fitness: 全局最大适应度值 
        population: 种群
        fits: 该种群对应的适应度值列表
    returns:
        cni: 更新后的无显著改善计数
        lsf: 更新后上一代总适应度值
        lbi: 更新后局部最有个体
        lbf: 更新后局部最大适应度值
        gbi: 更新后全局最优个体
        gbf: 更新后全局最大适应度值 
    '''
    s=sum(fits) # 适应度总值
    if s<last_sum_fitness*(no_improve_threshold+1): # 改进不明显
        cni=count_no_improve+1 # 增加计数
    else: # 改进明显
        cni=0 # 重置计数
    lsf=s # 更新上一代总适应度值
    lbf=max(fits) # 局部最大适应度
    lbi = individual_decoding(population[fits.index(lbf)] ) # 局部最佳个体
    if lbf>gloabal_best_fitness: # 如果大于全局最优，则更新
        gbf=lbf
        gbi=lbi
    else: # 否则不更新
        gbf=gloabal_best_fitness
        gbi=global_best_individual
    return cni,lsf,lbi,lbf,gbi,gbf

def save_generation(generation_index,population,fits):
    '''
    存储每一代的结果
    params:
        generation_index: 代数索引
        population: 种群
        fits: 适应度值列表
    '''
    path=os.path.join(result_path,f'generation_{generation_index}')
    os.makedirs(path)
    with open(os.path.join(path,'population.txt'),"w") as f:
        for i in range(len(population)):
            for j in population[i]:
                f.write(str(j)+" ")
            f.write("--> "+str(fits[i])+'\n')
    f.close()
    with open(os.path.join(path,'population.pkl'), 'wb') as f:
            pickle.dump(population, f)
    fig, ax = plt.subplots(dpi=200)
    ax.boxplot(fits,showfliers=False,vert=True)
    ax.set_title(f'{generation_index}代种群适应度箱型图')
    ax.set_ylabel('fitness')
    plt.savefig(os.path.join(path,'fitness_box.png'))
    plt.close()

def save_params(generation_index,local_best_individual,local_best_fitness,local_sum_fitness,global_best_individual,global_best_fitness):
    '''
    保存关键参数
    params:
        generation_index: 代数索引
        local_best_individual: 局部最优个体
        local_best_fitness: 局部最大适应值
        local_sum_fitness: 本代总适应值
        global_best_individual: 全局最优个体
        global_best_fitness: 全局最大适应值
    '''
    global result
    str_lbi='[' + ', '.join([str(t) for t in local_best_individual]) + ']'
    str_gbi='[' + ', '.join([str(t) for t in global_best_individual]) + ']'
    result=pd.concat([result,
                    pd.DataFrame({
                        'generation index':generation_index,
                        'local best individual':str_lbi, 
                        'local best fitness':local_best_fitness,
                        'local sum fitness':local_sum_fitness,
                        'global best individual':str_gbi,
                        'global best fitness':global_best_fitness
                        },index=[generation_index]
                    )])
    if generation_index%10==0:
        result.to_csv(result_path+'\\log.csv',index=False)
        plt.plot(result['generation index'].to_numpy(), result['local sum fitness'].to_numpy())
        plt.xlabel('generation index')
        plt.ylabel('sum of fitness')
        plt.title('总适应度值-种群代数 折线图')
        plt.savefig(os.path.join(result_path,'trend.png'))
        plt.close()

def save_origin(solution_num,population_size,generations,max_no_improve,no_improve_threshold,crossover_rate,mutation_rate):
    '''
    保存实验初始参数
    params:
        solution_num: 求解站点数目
        population_size: 种群规模
        generations: 最大迭代次数
        max_no_improve: 最大没有显著提升代数
        no_improve_threshold: 没有显著提升阈值
        crossover_rate:交叉概率
        mutation_rate:变异概率
    '''
    with open(result_path+'\\params.txt','w',encoding="utf-8") as f:
        f.write('求解站的个数: {}\n'.format(solution_num))
        f.write('种群规模: {}\n'.format(population_size))
        f.write('最大迭代次数: {}\n'.format(generations))
        f.write('种群不显著提升最大代数: {}\n'.format(max_no_improve))
        f.write('种群不显著提升阈值: {}\n'.format(no_improve_threshold))
        f.write('交叉概率: {}\n'.format(crossover_rate))
        f.write('变异概率: {}\n'.format(mutation_rate))
    f.close()

def GA(solution_num,population_size=200,generations=1000,max_no_improve=400,no_improve_threshold=0.0001,crossover_rate=0.9,mutation_rate=0.05):
    '''
    遗传算法求解
    params:
        solution_num: 求解观测站个数
        population_size: 种群规模
        generations: 最大迭代次数
        max_no_improve: 最多没有显著提升的代数阈值
        no_improve_threshold: 没有显著提升适应度比例阈值
        crossover_rate: 交叉概率
        mutation_rate: 变异概率
    returns:
        global_best_individual: 全局最优个体
        global_best_fitness: 全局最优个体的适应度值
    '''
    save_origin(solution_num,population_size,generations,max_no_improve,no_improve_threshold,crossover_rate,mutation_rate)
    count_no_improve=0 # 没有显著提升适应度值的代数
    last_sum_fitness=0 # 上一轮种群总适应度值
    local_best_individual=None # 局部最优个体
    local_best_fitness=0 # 局部最大适应度值
    global_best_individual=None # 全局最优个体
    global_best_fitness=0 # 全局最大适应度值
    print('-->初始化种群')
    population=generate_population(solution_num,population_size) # 生成初始化种群
    fits= [fitness_function(chromosome) for chromosome in population] # 计算适应度列表
    save_generation(0,population,fits)
    # 更新参数
    count_no_improve,last_sum_fitness,local_best_individual,local_best_fitness,global_best_individual,global_best_fitness=\
        calLocal_updateGlobal(no_improve_threshold,count_no_improve,last_sum_fitness,global_best_individual,global_best_fitness,population,fits)
    print(f"-->Generation: 0, Total Fitness: {sum(fits)}, Local Best Fitness: {local_best_fitness}, Global Best Fitness: {global_best_fitness}")
    save_params(0,local_best_individual,local_best_fitness,sum(fits),global_best_individual,global_best_fitness)
    print('-->开始迭代')
    for generation in range(generations):
        next_population=[] # 下一代种群
        # 父体选择
        father_population = selection(population,fits,population_size) # 父代
        num_of_father=int(len(father_population)*crossover_rate) if int(len(father_population)*crossover_rate)%2==0\
            else int(len(father_population)*crossover_rate)-1 # 根据交叉概率确定重组的父体个数
        # 交叉重组
        father_population=random.sample(father_population,num_of_father) # 随机抽取父体中的染色体进行重组
        child_population = [] # 孩子种群
        for i in range(0,len(father_population),2):
            # 交叉
            parent1, parent2 = father_population[i],father_population[i+1]
            child1, child2 = crossover(parent1, parent2)
            child_population.extend([child1, child2])
        # 变异
        population.extend(child_population) # 将原来的种群和现在产生的孩子合在一起
        for individual in population:
            individual = mutation(individual,mutation_rate) # 变异
            # 加入下一代种群
            next_population.append(individual)
        # 存活选择
        next_population=sorted(next_population,key=fitness_function,reverse=True)[:population_size]
        population = next_population # 更新种群
        fits= [fitness_function(chromosome) for chromosome in population] # 计算适应度列表
        save_generation(generation+1,population,fits)
        # 更新参数
        count_no_improve,last_sum_fitness,local_best_individual,local_best_fitness,global_best_individual,global_best_fitness=\
        calLocal_updateGlobal(no_improve_threshold,count_no_improve,last_sum_fitness,global_best_individual,global_best_fitness,population,fits)
        save_params(generation+1,local_best_individual,local_best_fitness,sum(fits),global_best_individual,global_best_fitness)
        print(f"-->Generation: {generation+1}, Total Fitness: {sum(fits)}, Local Best Fitness: {local_best_fitness}, Global Best Fitness: {global_best_fitness}")
        if count_no_improve>=max_no_improve: # 判断是否收敛
            break
    if generation==generations:
        print('##>达到最大迭代次数，算法结束')
    else:
        print(f'##>算法已经{max_no_improve}代没有明显改进，算法结束')
    print('~~>问题最优解为: {}, 其适应值为: {}'.format(global_best_individual,global_best_fitness))
    return global_best_individual,global_best_fitness

def RecToGeo(i,j):
    '''
    矩阵行列号转经纬度
    params:
        i: 矩阵行号
        j: 矩阵列号
    returns:
        lat: 纬度
        lon: 经度
    '''
    # 获取站点所在的大格子
    big_point=getRightBigPoint((i,j))
    # 计算在每个大格子里面站点的索引
    sub_i=i%10-4
    sub_j=j%10-4
    # 通过大格子中心的经纬度计算站点经纬度
    lat=big_point.lat-sub_i*0.01
    lon=big_point.lon+sub_j*0.01
    return round(lat,2),round(lon,2)

def result_postproccess(global_best_individual,global_best_fitness):
    '''
    结果后处理
    '''
    # 输出最终结果
    pt=prettytable.PrettyTable()
    pt.field_names=['站点序号','经度','纬度','矩形坐标']
    index=0
    for point in global_best_individual:
        index+=1
        lat,lon=RecToGeo(point[0],point[1])
        pt.add_row([index,lon,lat,point])
    with open(result_path+"\\result.txt", 'w',encoding="utf-8") as f:
        f.write(pt.get_string())
        f.write(f"\n观测效力: {global_best_fitness}")
    f.close()
    # 绘制点位图
    lat_max,lon_min=RecToGeo(0,0)
    lat_min,lon_max=RecToGeo(69,69)
    plt.figure(figsize=(8, 6))# 设置图形大小
    plt.grid(True)# 绘制网格
    plt.xlim(lon_min, lon_max)# 设置 x 轴范围
    plt.ylim(lat_min, lat_max)# 设置 y 轴范围
    plt.title('最佳站点分布图')
    for point in global_best_individual:
        lat,lon=RecToGeo(point[0],point[1])
        plt.scatter(lon,lat, color='red', marker='o')
    plt.savefig(result_path+"\\points.png")
    plt.close()
    # 计算并展示效力矩阵
    values={}
    for point in global_best_individual:
        big_point=getRightBigPoint(point)
        values=calY(big_point,point,values)
    min_row = min(key[0] for key in values) - 1
    min_col = min(key[1] for key in values) - 1
    max_row = max(key[0] for key in values) + 1
    max_col = max(key[1] for key in values) + 1
    arr = np.zeros((max_row-min_row, max_col-min_col))
    for key, value in values.items():
        arr[key[0]-min_row, key[1]-min_col] = value
    np.savetxt(result_path+'\\sum_H.csv', arr, delimiter=',')
    plt.imshow(arr, cmap='viridis',interpolation='nearest')
    plt.colorbar()  # 添加颜色条
    plt.axis('off')  # 去掉坐标轴
    plt.title('最佳站点观测效力')
    plt.savefig(result_path+"\\sum_H.png")
    plt.close()



solution_num=3

print('读取甲烷通量场tif文件')
data,origin_lat,origin_lon=readXTifFile()
print('对甲烷通量场进行插值')
interp_func = interp2d(np.arange(data.shape[1]), np.arange(data.shape[0]), data, kind='linear') # 创建插值函数
data = interp_func(np.arange(data.shape[1]*10) / 10, np.arange(data.shape[0]*10) / 10) # 在目标尺寸上进行插值
print('-->插值后 高:',data.shape[0],' 宽:',data.shape[1])
# 存储插值结果
if not os.path.exists('实验一\\interpolation_data.csv'):
    np.savetxt('实验一\\interpolation_data.csv', data, delimiter=',')
# 绘制热力图
plt.imshow(data[1500:2500,3500:4500], cmap='viridis', interpolation='nearest')
plt.colorbar()  # 添加颜色条
plt.axis('off')  # 去掉坐标轴
plt.title('插值后甲烷通量场')
plt.savefig("实验一\\interpolation_data.png")
plt.close()
print('生成小格点数据矩阵')
SP=generateSmallPoints(data,origin_lat,origin_lon)
print('读取足迹矩阵csv文件,并生成大格点H矩阵')
BP=readHCsvFile()
print('遗传算法求解')
result= pd.DataFrame(columns=['generation index', 'local best individual', 'local best fitness', 'local sum fitness', 'global best individual','global best fitness'])
result_path=f'D:\\本科\\空间智能计算与服务\\实习-mwz\\实验一\\result-{solution_num}\\result'
i=1
while os.path.exists(result_path+str(i)):
    i+=1
result_path=result_path+str(i)
os.makedirs(result_path)
global_best_individual,global_best_fitness=GA(solution_num=solution_num)
result_postproccess(global_best_individual,global_best_fitness)
