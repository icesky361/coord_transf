#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
高德坐标到思极坐标转换推理工具

该脚本用于加载已训练的坐标转换模型，并批量将Excel文件中的高德坐标转换为思极坐标。
支持自动识别坐标列和手动选择列功能，适用于处理各种格式的坐标数据。
"""
import os
import sys
import pandas as pd  # 用于Excel文件处理和数据操作
import joblib        # 用于模型加载和保存
import numpy as np   # 用于数值计算

# 确保中文显示正常
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class SJInferenceConverter:
    """
    高德坐标系到思极坐标系推理转换工具类
    
    该类实现了加载已训练的模型，并使用模型进行高德坐标到思极坐标的批量转换。
    支持从Excel文件读取数据并将转换结果保存到新的Excel文件中。
    
    主要工作流程：
    1. 加载训练好的模型
    2. 读取Excel文件中的坐标数据
    3. 识别高德经纬度列
    4. 使用模型进行坐标转换
    5. 将转换结果保存到新的Excel文件
    """
    def __init__(self):
        """初始化推理转换器实例"""
        self.model = None  # 存储加载的模型
        self.base_path = self._get_base_path()  # 获取程序基准路径

    def _get_base_path(self):
        """
        获取程序的基准路径
        
        Returns:
            str: 程序所在目录的绝对路径
        """
        if getattr(sys, 'frozen', False):
            # 如果是打包后的可执行文件
            return os.path.dirname(sys.executable)
        else:
            # 如果是Python脚本
            return os.path.dirname(os.path.abspath(__file__))

    def load_model(self, model_path=None):
        """
        加载已训练的坐标转换模型
        
        Args:
            model_path (str, optional): 模型文件路径，如果为None则使用默认路径
        
        Returns:
            bool: 加载成功返回True，否则返回False
        """
        if model_path is None:
            model_path = os.path.join(self.base_path, 'coordinate_model.pkl')

        try:
            if not os.path.exists(model_path):
                print(f"错误: 模型文件不存在 - {model_path}")
                return False

            self.model = joblib.load(model_path)
            print(f"成功加载模型: {model_path}")
            return True
        except Exception as e:
            print(f"加载模型时出错: {str(e)}")
            return False

    def find_gaode_columns(self, df):
        """
        智能识别高德坐标列
        
        Args:
            df (pd.DataFrame): 包含坐标数据的数据框
        
        Returns:
            tuple: (lng_col, lat_col) 经度列名和纬度列名，如果未找到则返回(None, None)
        """
        # 定义高德坐标的关键词
        gaode_keywords = ['高德', 'gcj02', 'gcj', '02']
        lng_keywords = ['经度', 'longitude', 'lng']
        lat_keywords = ['纬度', 'latitude', 'lat']

        # 查找经度列
        lng_col = None
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword.lower() in col_lower for keyword in gaode_keywords) and \
               any(lng_key in col_lower for lng_key in lng_keywords):
                lng_col = col
                break

        # 查找纬度列
        lat_col = None
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword.lower() in col_lower for keyword in gaode_keywords) and \
               any(lat_key in col_lower for lat_key in lat_keywords):
                lat_col = col
                break

        # 如果没有找到带高德关键词的列，尝试查找通用的经纬度列
        if lng_col is None:
            for col in df.columns:
                col_lower = col.lower()
                if any(lng_key in col_lower for lng_key in lng_keywords):
                    lng_col = col
                    break

        if lat_col is None:
            for col in df.columns:
                col_lower = col.lower()
                if any(lat_key in col_lower for lat_key in lat_keywords):
                    lat_col = col
                    break

        return lng_col, lat_col

    def convert_coordinates(self, df, lng_col, lat_col):
        """
        批量转换坐标
        
        Args:
            df (pd.DataFrame): 包含坐标数据的数据框
            lng_col (str): 经度列名
            lat_col (str): 纬度列名
        
        Returns:
            pd.DataFrame: 包含转换结果的数据框，添加了'思极经度'和'思极纬度'列
        """
        # 提取高德坐标
        gaode_coords = df[[lng_col, lat_col]].values.astype(np.float64)

        # 预测偏移量 - 模型返回的是思极坐标相对于高德坐标的偏移值
        offsets = self.model.predict(gaode_coords)

        # 计算思极坐标 = 高德坐标 + 偏移量
        sj_lng = gaode_coords[:, 0] + offsets[:, 0]
        sj_lat = gaode_coords[:, 1] + offsets[:, 1]

        # 添加到数据框
        df['思极经度'] = sj_lng
        df['思极纬度'] = sj_lat

        return df

    def process_excel_file(self, input_file, output_file):
        """
        处理Excel文件，进行坐标转换
        
        Args:
            input_file (str): 输入Excel文件路径
            output_file (str): 输出Excel文件路径
        
        Returns:
            bool: 处理成功返回True，否则返回False
        """
        try:
            # 读取Excel文件
            print(f"正在读取文件: {input_file}")
            df = pd.read_excel(input_file, engine='openpyxl')
            print(f"成功读取文件，共{len(df)}条记录")

            # 识别高德坐标列
            lng_col, lat_col = self.find_gaode_columns(df)
            if lng_col is None or lat_col is None:
                print("错误: 无法识别高德经纬度列")
                return False

            print(f"识别到高德坐标列: 经度='{lng_col}', 纬度='{lat_col}'")

            # 进行坐标转换
            df = self.convert_coordinates(df, lng_col, lat_col)
            print("坐标转换完成")

            # 确保输出目录存在
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # 保存结果
            df.to_excel(output_file, index=False, engine='openpyxl')
            print(f"结果已保存到: {output_file}")
            return True
        except Exception as e:
            print(f"处理文件时出错: {str(e)}")
            return False

    def main(self):
        """
        主函数，提供命令行交互界面
        
        引导用户选择Excel文件，然后调用相关方法进行坐标转换
        """
        print("=====================")
        print("  高德坐标转思极坐标  ")
        print("        推理工具      ")
        print("=====================")

        # 加载模型
        if not self.load_model():
            print("无法加载模型，程序退出")
            return

        # 列出当前目录下的Excel文件
        excel_files = [f for f in os.listdir(self.base_path) if f.endswith('.xlsx')]
        if not excel_files:
            print("错误: 当前目录下没有找到Excel文件(.xlsx)")
            return

        # 显示文件列表
        print("\n当前目录下的Excel文件:")
        for i, file in enumerate(excel_files, 1):
            print(f"  {i}. {file}")

        # 让用户选择文件
        while True:
            try:
                choice = input("\n请选择要处理的文件序号(1-{}): ".format(len(excel_files)))
                choice = int(choice)
                if 1 <= choice <= len(excel_files):
                    input_filename = excel_files[choice - 1]
                    break
                else:
                    print(f"请输入1到{len(excel_files)}之间的数字")
            except ValueError:
                print("请输入有效的数字")

        # 构建输入和输出文件路径
        input_file = os.path.join(self.base_path, input_filename)
        filename_without_ext = os.path.splitext(input_filename)[0]
        output_filename = f"{filename_without_ext}-思极坐标.xlsx"
        output_file = os.path.join(self.base_path, output_filename)

        # 处理文件
        self.process_excel_file(input_file, output_file)

        # 等待用户按键退出
        input("\n按回车键退出...")

if __name__ == "__main__":
    # 创建转换器实例并运行主函数
    converter = SJInferenceConverter()
    converter.main()