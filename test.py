
# These two lines are necessary only if GemPy is not installed
import sys, os
sys.path.append("C:\ProgramData\Anaconda3\Lib\site-packages\pip")
sys.path.append("C:\ProgramData\Anaconda3\Lib\site-packages\pip\_vendor")
sys.path.append("C:\ProgramData\Anaconda3\Lib\site-packages\pip\_vendor\pkg_resources")
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Embedding matplotlib figures in the notebooks
#%matplotlib inline

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

# Importing auxiliary libraries
import numpy as np
import matplotlib.pyplot as plt


# Importing GemPy
import gempy as gp
# Importing the data from CSV-files and setting extent and resolution
#geo_data = gp.create_data([0,2000,0,2000,0,2000],[100,100,100],
#                          path_o = os.pardir+"/gempy/input_data/tut_chapter1/simple_fault_model_orientations.csv", # importing orientation (foliation) data
#                          path_i = os.pardir+"/gempy/input_data/tut_chapter1/simple_fault_model_points.csv") # importing point-positional interface data
####修改后的路径
# Importing the data from CSV-files and setting extent and resolution
geo_data = gp.create_data([0,2000,0,2000,0,2000],[100,100,100],
                          path_o = "C:/gempy-1.0/cgre-aachen-gempy-59582d1/notebooks/input_data/tut_chapter1/simple_fault_model_orientations.csv", # importing orientation (foliation) data
                          path_i = "C:/gempy-1.0/cgre-aachen-gempy-59582d1/notebooks/input_data/tut_chapter1/simple_fault_model_points.csv") # importing point-positional interface data


#####列出输入数据
gp.get_data(geo_data)#列出输入数据



#########
# Assigning series to formations as well as their order (timewise)
#gp.set_series(geo_data, {"Fault_Series":'Main_Fault',
#                      "Strat_Series": ('Sandstone_2','Siltstone', 'Shale', 'Sandstone_1')},
#                       order_series = ["Fault_Series", 'Strat_Series'],
#                       order_formations=['Main_Fault',
#                                         'Sandstone_2','Siltstone', 'Shale', 'Sandstone_1',
#                                         ], verbose=0)

# unconformity model:
gp.set_series(geo_data, {"Fault_Series":'Main_Fault', "Unconf_Series":'Carbonate',
                      "Strat_Series": ('Sandstone_2','Siltstone', 'Shale', 'Sandstone_1')},
                       order_series = ["Fault_Series", "Unconf_Series", 'Strat_Series'],
                       order_formations=['Main_Fault', 'Carbonate',
                                         'Sandstone_2','Siltstone', 'Shale', 'Sandstone_1',
                                         ], verbose=0)

#%matplotlib inline
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
gp.get_sequential_pile(geo_data)

print(gp.get_grid(geo_data))


#We introduced the function get_data above. You can also specify which kind of data you want to call,
# by declaring the string attribute “dtype” to be either “interfaces” (surface points) or “foliations” (orientation measurements).
#gp.get_data(geo_data, 'interfaces').head()
#gp.get_data(geo_data, 'orientations')


#使用函数plot_data，我们获得了数据点在选定方向的平面上的2D投影
#%matplotlib inline
#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')
#gp.plot_data(geo_data, direction='y')


#打开一个带有我们数据的3D交互式绘图的新窗口
#gp.plot_data_3D(geo_data)


#通过以下函数从对象中生成一个InterpolatorInput对象（interp_data在这些教程中命名 ）完成的InputData
interp_data = gp.InterpolatorData(geo_data, u_grade=[1,1], output='geology', compile_theano=True, theano_optimizer='fast_compile')
print(interp_data)

#可以找出哪个编号已被分配到哪个编队：
#interp_data.geo_data_res.get_formation_number()

#用于插补的参数可以使用该函数返回get_kriging_parameters。
#gp.get_kriging_parameters(interp_data) # Maybe move this to an extra part?

#我们需要通过计算我们的完整模型 compute_model。默认情况下，这将以数组的形式返回两个单独的解决方案。
#第一个给出了岩性地层的信息，第二个在模型中的断层网络上。这些数组由两个子数组组成，每个数组都有：
#岩性块模型解决方案：
#入口[0]：这个数组显示在每个体素中发现了什么样的岩性地层，如相应地层编号所示。
#条目[1]：代表块模型中岩性单元和层的方向的势场阵列。
#故障网络模块解决方案：
#条目[0]：模型中所有由断层分隔的区域都由每个体素中包含的不同数字表示的数组。
#条目[1]：块模型中与故障网络相关的潜在磁场阵列。
lith_block, fault_block = gp.compute_model(interp_data)

