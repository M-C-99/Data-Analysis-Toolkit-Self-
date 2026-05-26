import pandas as pd
import numpy as np
import os
from typing import Union, List, Dict, Optional


class DataPreprocessor:
    def __init__(self, path: str, **kwargs):
        # 导入的对象为文件的地址
        self.path = path
        # 原始数据
        # Optional[pd.DaraFrame] 意为可以为DataFrame格式，初始为None
        self.raw_data: Optional[pd.DataFrame] = None
        # 更改为DataFrame格式的文件【便于处理】
        # Optional[pd.DaraFrame] 意为可以为DataFrame格式，初始为None
        self.df: Optional[pd.DataFrame] = None
    # -------------------------------------------------------------------------------------------------------------

    def upload_data(self, file_type: str = None, **kwargs) -> pd.DataFrame:
        # file_type: 可以显式给出文件类型，也可以自动判断
        # **kwargs: 常用的传入pandas的参数，
        # 例如：暂时不设置列名：header=None(header参数默认是header=0设置第一行为列名)
        # 关于header，如果设置header=2，那就意味着跳过前两行，用第三行做列名，下面的数据保持
        # 首先检查是否找到目标文件：
        if not os.path.exists(self.path):
            raise FileNotFoundError(f'未找到文件：{self.path}')
        # 然后判断给出文件的类型：
        # 如果后缀没有显式输入，则自动判断类型
        if file_type == None:
            _, hou_zhui = os.path.splitext(self.path)
            file_type = hou_zhui.lstrip('.').lower()
        if file_type == 'xls':
            self.df = pd.read_excel(self.path, **kwargs)
        elif file_type == 'xlsx':
            self.df = pd.read_excel(self.path, **kwargs)
        elif file_type == 'csv':
            self.df = pd.read_csv(self.path, **kwargs)
        elif file_type == 'json':
            self.df = pd.read_json(self.path, **kwargs)
        elif file_type == "parquet":
            self.df = pd.read_parquet(self.path, **kwargs)
        else:
            raise ValueError(f'不支持此类型，可以支持的类型有【xls, xlsx, csv, json, parquet】')

        print(f'导入成功，文件共有{self.df.shape[0]}行, {self.df.shape[1]}列')
        return self.df

    # -------------------------------------------------------------------------------------------------------------

    def handle_missing_values(self, method: str = 'drop', columns: list = None, fill_value=0) -> pd.DataFrame:
        # 处理策略 - 'drop'(删除), 'mean'(均值填充), 'median'(中位数填充), 'mode'(众数填充), 'constant'(固定值填充)
        # 留存原表格：
        raw_df = self.df.copy()
        # 检测是否数据表正确导入
        if self.df is None:
            raise ValueError('没有检测到数据表，请导入')
        # 拿出需要处理的列(也是后续做分析的列)，如果没有输入特定的列，那就是全局处理
        if columns is not None:
            handle_df = self.df[columns]
        else:
            handle_df = self.df
        # 选择处理方式：
        # 1. 直接drop：
        if method == 'drop':
            self.df = self.df.dropna(subset=columns)
        # 2. 用当列的平均值填充
        elif method == 'mean':
            # 筛选数值列进行填充
            numeric_cols = self.df.select_dtypes(include='number').columns
            if len(numeric_cols) == 0:
                raise ValueError('在所选列中没有数值列，无法用均值填充')
            mean_values = self.df[numeric_cols].mean()
            self.df[numeric_cols] = self.df[numeric_cols].fillna(mean_values)
        # 3. 用当列的中位数进行填充
        elif method == 'median':
            # 筛选数值列进行填充
            numeric_cols = self.df.select_dtypes(include='number').columns
            if len(numeric_cols) == 0:
                raise ValueError('在所选列中没有数值列，无法用均值填充')
            median_values = self.df[numeric_cols].median()
            self.df[numeric_cols] = self.df[numeric_cols].fillna(median_values)
        # 4. 用当列的众数进行填充
        elif method == 'mode':
            # 多众数则选择第一个
            mode_values = self.df.mode().iloc[0]
            self.df = self.df.fillna(mode_values)
        # 5. 填充常数，在使用方法的时候会填入，通常填入0，这里的默认值也为0
        elif method == 'constant':
            self.df = self.df.fillna(fill_value)

        else:
            raise ValueError(
                f"不支持的处理方法: {method}\n"
                f"支持的方法有: 'drop', 'mean', 'median', 'mode', 'constant'"
            )

        return self.df

    # -------------------------------------------------------------------------------------------------------------

    def handle_repeat_values(self, keep: str = 'first'):
        if self.df is None:
            raise ValueError('没有检测到数据表，请导入')
        # 去除所有列完全相同的重复行。
        # :param df: 输入 DataFrame
        # :param keep: 保留规则
        #    'first' - 保留第一次出现的行（默认）
        #    'last'  - 保留最后一次出现的行
        #     False   - 删除所有重复行（一条都不留）
        # :return: 去重后的 DataFrame（新对象，原 df 不变）

        return self.df.drop_duplicates(keep=keep)

    # -------------------------------------------------------------------------------------------------------------

    def sort_values(self, by: list, ascending: bool = True) -> pd.DataFrame:
        # 对表格进行指定col/cols的排序
        # ascending=True --> 升序；ascending=False --> 降序
        if self.df is None:
            raise ValueError("数据尚未导入，请先调用 upload_data()。")

        self.df = self.df.sort_values(by=by, ascending=ascending)

        return self.df

    # -------------------------------------------------------------------------------------------------------------

    def set_columns_name(self, column_names: list) -> pd.DataFrame:
        # 输入每一列的名称
        # 在upload_data()函数中，如果输入自带参数"header=None", 则需要自己输入列名
        if self.df is None:
            raise ValueError("数据尚未导入，请先调用 load_data()。")
        if len(column_names) != len(self.df.columns):
            raise ValueError(
                f"新列名数量 ({len(column_names)}) 与原数据列数 ({len(self.df.columns)}) 不匹配。"
            )

        self.df.columns = column_names
        return self.df

    # -------------------------------------------------------------------------------------------------------------

    def set_index(self, column_name: str, drop: bool = True) -> pd.DataFrame:
        # 设置某一列为index
        # drop意味设置为index的列是否在df格式数据中被去除
        if self.df is None:
            raise ValueError('数据尚未导入，请先调用upload_data()函数进行数据载入')

        if column_name not in self.df.columns:
            raise KeyError(f"列名 '{column_name}' 不存在于数据中。")

        self.df = self.df.set_index(column_name, drop=drop)
        return self.df

    # -------------------------------------------------------------------------------------------------------------

    def set_columns_dtype(self, column_dtype: dict) -> pd.DataFrame:
        # 设置指定列的数据类型。
        #
        # :param col_dtype: 字典，键为列名，值为目标数据类型，
        #                   例如 {'age': 'int64', 'salary': 'float', 'join_date': 'datetime64'}
        # :return: self.df（支持链式调用）
        if self.df is None:
            raise ValueError("数据尚未导入，请先调用 upload_data()。")

        for col, dtype in column_dtype.items():
            if col not in self.df.columns:
                raise KeyError(f"列名 '{col}' 不存在于数据中。")
            try:
                self.df[col] = self.df[col].astype(dtype)
            except Exception as e:
                raise ValueError(f"无法将列 '{col}' 转换为类型 {dtype}，错误详情: {e}")

        return self.df


