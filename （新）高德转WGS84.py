import os
import sys
import pandas as pd
import math

# 定义 gcj02_to_wgs84 函数
def transformlat(x, y):
    ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(y * math.pi) + 40.0 * math.sin(y / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(y / 12.0 * math.pi) + 320 * math.sin(y * math.pi / 30.0)) * 2.0 / 3.0
    return ret

def transformlng(x, y):
    ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(x * math.pi) + 40.0 * math.sin(x / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(x / 12.0 * math.pi) + 300.0 * math.sin(x / 30.0 * math.pi)) * 2.0 / 3.0
    return ret

def out_of_china(lng, lat):
    return not (72.004 <= lng <= 137.8347 and 0.8293 <= lat <= 55.8271)

def gcj02_to_wgs84(lng, lat):
    a = 6378245.0  # 地球长半轴
    ee = 0.00669342162296594323  # 扁率
    if out_of_china(lng, lat):
        return [lng, lat]
    dlat = transformlat(lng - 105.0, lat - 35.0)
    dlng = transformlng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * math.pi
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * math.pi)
    dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * math.pi)
    mglat = lat + dlat
    mglng = lng + dlng
    return [lng * 2 - mglng, lat * 2 - mglat]

def batch_convert(input_file, output_file):
    # 读取输入文件
    df = pd.read_excel(input_file, engine='openpyxl')
    
    # 执行坐标转换
    df[['经度（WGS84计算）', '纬度（WGS84计算）']] = df.apply(
        lambda row: gcj02_to_wgs84(row['经度（高德）'], row['纬度（高德）']),
        axis=1,
        result_type='expand'
    )
    
    # 找到纬度（高德）列的索引
    lat_gd_index = df.columns.get_loc('纬度（高德）')
    
    # 重新排列列顺序
    new_columns = df.columns[:lat_gd_index + 1].tolist() + ['经度（WGS84计算）', '纬度（WGS84计算）'] + df.columns[lat_gd_index + 1:-2].tolist()
    df = df[new_columns]
    
    # 检查输出文件所在目录是否存在，如果不存在则创建
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 检查输出文件是否存在
    if os.path.exists(output_file):
        mode = 'a'
        if_sheet_exists = 'replace'
    else:
        mode = 'w'
        if_sheet_exists = None
    
    # 保存结果到新Excel文件
    with pd.ExcelWriter(output_file, engine='openpyxl', mode=mode, if_sheet_exists=if_sheet_exists) as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')

# 获取可执行文件的路径
if getattr(sys, 'frozen', False):
    # 如果是打包后的可执行文件
    base_path = os.path.dirname(sys.executable)
else:
    # 如果是Python脚本
    base_path = os.path.dirname(os.path.abspath(__file__))

# 配置参数
input_path = os.path.join(base_path, '监拍经纬度.xlsx')
output_path = os.path.join(base_path, '监拍经纬度output.xlsx')

# 配置日志记录
import logging
logging.basicConfig(filename='app.log', level=logging.DEBUG)

# 在关键位置添加日志记录
logging.debug(f"Input file path: {input_path}")
logging.debug(f"Output file path: {output_path}")

# 执行转换
batch_convert(input_path, output_path)