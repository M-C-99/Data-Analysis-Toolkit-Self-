from general import *
import pandas as pd
import numpy as np

# 数据预处理：
emp_test_table = DataPreprocessor(path='/Users/suncheng/PycharmProjects/数据分析工具库/员工信息测试表.xlsx')
emp_test_table.upload_data(header=1)
emp_test_table.handle_missing_values(method='constant', fill_value=0)
emp_test_table.sort_values(by=['基本工资(元)'])
emp_test_table.set_index(column_name='序号')

df = emp_test_table.df

# 数据

