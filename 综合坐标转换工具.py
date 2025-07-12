import os
import sys
import pandas as pd
import math
import re

# 定义坐标转换所需的通用函数
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

# WGS84坐标系转高德坐标系(GCJ-02)
def wgs84_to_gcj02(lng, lat):
    """
    WGS84坐标系转高德坐标系(GCJ-02)
    :param lng: WGS84坐标系下的经度
    :param lat: WGS84坐标系下的纬度
    :return: 高德坐标系下的经纬度
    """
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
    return [mglng, mglat]  # 返回高德坐标

# 高德坐标系(GCJ-02)转WGS84坐标系
def gcj02_to_wgs84(lng, lat):
    """
    高德坐标系(GCJ-02)转WGS84坐标系
    :param lng: 高德坐标系下的经度
    :param lat: 高德坐标系下的纬度
    :return: WGS84坐标系下的经纬度
    """
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
    return [lng * 2 - mglng, lat * 2 - mglat]  # 返回WGS84坐标

# GCJ02坐标系转百度坐标系(BD09)
def gcj02_to_bd09(lng, lat):
    """
    GCJ02坐标系转百度坐标系(BD09)
    :param lng: GCJ02坐标系下的经度
    :param lat: GCJ02坐标系下的纬度
    :return: 百度坐标系下的经纬度
    """
    z = math.sqrt(lng * lng + lat * lat) + 0.00002 * math.sin(lat * math.pi * 3000.0 / 180.0)
    theta = math.atan2(lat, lng) + 0.000003 * math.cos(lng * math.pi * 3000.0 / 180.0)
    bd_lng = z * math.cos(theta) + 0.0065
    bd_lat = z * math.sin(theta) + 0.006
    return [bd_lng, bd_lat]  # 返回百度坐标

# 百度坐标系(BD09)转GCJ02坐标系
def bd09_to_gcj02(lng, lat):
    """
    百度坐标系(BD09)转GCJ02坐标系
    :param lng: 百度坐标系下的经度
    :param lat: 百度坐标系下的纬度
    :return: GCJ02坐标系下的经纬度
    """
    x = lng - 0.0065
    y = lat - 0.006
    z = math.sqrt(x * x + y * y) - 0.00002 * math.sin(y * math.pi * 3000.0 / 180.0)
    theta = math.atan2(y, x) - 0.000003 * math.cos(x * math.pi * 3000.0 / 180.0)
    gcj_lng = z * math.cos(theta)
    gcj_lat = z * math.sin(theta)
    return [gcj_lng, gcj_lat]  # 返回GCJ02坐标

# WGS84坐标系转百度坐标系(BD09)
def wgs84_to_bd09(lng, lat):
    """
    WGS84坐标系转百度坐标系(BD09)
    :param lng: WGS84坐标系下的经度
    :param lat: WGS84坐标系下的纬度
    :return: 百度坐标系下的经纬度
    """
    gcj02 = wgs84_to_gcj02(lng, lat)
    return gcj02_to_bd09(gcj02[0], gcj02[1])

# 百度坐标系(BD09)转WGS84坐标系
def bd09_to_wgs84(lng, lat):
    """
    百度坐标系(BD09)转WGS84坐标系
    :param lng: 百度坐标系下的经度
    :param lat: 百度坐标系下的纬度
    :return: WGS84坐标系下的经纬度
    """
    gcj02 = bd09_to_gcj02(lng, lat)
    return gcj02_to_wgs84(gcj02[0], gcj02[1])

# 智能识别列名函数
def find_column_by_keywords(df, type_keywords, coord_keyword):
    """
    智能识别列名
    :param df: DataFrame对象
    :param type_keywords: 坐标系类型关键词列表，如['WGS84', '84']
    :param coord_keyword: 坐标类型关键词，如'经度'或'纬度'
    :return: 匹配的列名或None
    """
    # 首先尝试精确匹配
    for col in df.columns:
        # 检查列名是否同时包含坐标系类型关键词和坐标类型关键词
        if any(keyword.lower() in col.lower() for keyword in type_keywords) and coord_keyword in col:
            return col
    
    # 如果没有找到精确匹配，尝试只匹配坐标类型关键词
    for col in df.columns:
        if coord_keyword in col:
            return col
    
    return None

# 批量转换函数
def batch_convert(input_file, output_file, convert_func, source_type, target_type):
    """
    批量转换坐标
    :param input_file: 输入文件路径
    :param output_file: 输出文件路径
    :param convert_func: 转换函数
    :param source_type: 源坐标系类型，如'WGS84'、'高德'或'百度'
    :param target_type: 目标坐标系类型，如'WGS84'、'高德'或'百度'
    """
    # 读取输入文件
    df = pd.read_excel(input_file, engine='openpyxl')
    
    # 定义坐标系关键词映射
    keywords_map = {
        'WGS84': ['WGS84', '84', 'wgs'],
        '高德': ['高德', 'GCJ02', 'gcj', '02'],
        '百度': ['百度', 'BD09', 'bd', '09']
    }
    
    # 查找源坐标系的经纬度列
    source_lng_col = find_column_by_keywords(df, keywords_map[source_type], '经度')
    source_lat_col = find_column_by_keywords(df, keywords_map[source_type], '纬度')
    
    if source_lng_col is None or source_lat_col is None:
        raise ValueError(f"无法找到{source_type}经纬度列，请确保Excel文件包含经度和纬度列")
    
    # 输出列名
    target_lng_col = f'经度（{target_type}计算）'
    target_lat_col = f'纬度（{target_type}计算）'
    
    # 执行坐标转换
    df[[target_lng_col, target_lat_col]] = df.apply(
        lambda row: convert_func(row[source_lng_col], row[source_lat_col]),
        axis=1,
        result_type='expand'
    )
    
    # 找到纬度列的索引
    lat_index = df.columns.get_loc(source_lat_col)
    
    # 重新排列列顺序
    new_columns = df.columns[:lat_index + 1].tolist() + [target_lng_col, target_lat_col] + df.columns[lat_index + 1:].tolist()
    # 移除可能的重复列
    new_columns = [col for i, col in enumerate(new_columns) if col not in new_columns[:i]]
    df = df[new_columns]
    
    # 检查输出文件所在目录是否存在，如果不存在则创建
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
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

# 配置日志记录
import logging
logging.basicConfig(filename=os.path.join(base_path, 'app.log'), level=logging.DEBUG)

# 主函数
def main():
    print("综合坐标转换工具")
    print("=" * 30)
    
    try:
        # 提示用户选择转换模式
        while True:
            print("请选择坐标转换模式：")
            print("1. WGS84 → 高德(GCJ-02)")
            print("2. 高德(GCJ-02) → WGS84")
            print("3. WGS84 → 百度(BD-09)")
            print("4. 百度(BD-09) → WGS84")
            print("5. 高德(GCJ-02) → 百度(BD-09)")
            print("6. 百度(BD-09) → 高德(GCJ-02)")
            mode = input("请选择(1-6): ")
            if mode in ['1', '2', '3', '4', '5', '6']:
                break
            print("输入无效，请重新输入！")
        
        # 列出当前目录下的Excel文件
        excel_files = [f for f in os.listdir(base_path) if f.endswith('.xlsx')]
        if excel_files:
            print("\n当前目录下的Excel文件:")
            for i, file in enumerate(excel_files, 1):
                print(f"{i}. {file}")
        else:
            print("\n当前目录下没有Excel文件！")
            return
        
        # 提示用户输入文件名
        while True:
            input_filename = input("\n请输入需处理的文件名(不含路径，例如：data.xlsx): ")
            if input_filename.endswith('.xlsx'):
                input_file = os.path.join(base_path, input_filename)
                if os.path.exists(input_file):
                    break
                else:
                    print(f"文件 {input_filename} 不存在，请重新输入！")
            else:
                print("请输入有效的Excel文件名(.xlsx)")
        
        # 根据模式设置转换函数和输出文件名
        filename_without_ext = os.path.splitext(input_filename)[0]
        
        # 设置转换参数
        conversion_params = {
            '1': {
                'func': wgs84_to_gcj02,
                'source': 'WGS84',
                'target': '高德',
                'suffix': '-高德'
            },
            '2': {
                'func': gcj02_to_wgs84,
                'source': '高德',
                'target': 'WGS84',
                'suffix': '-WGS84'
            },
            '3': {
                'func': wgs84_to_bd09,
                'source': 'WGS84',
                'target': '百度',
                'suffix': '-百度'
            },
            '4': {
                'func': bd09_to_wgs84,
                'source': '百度',
                'target': 'WGS84',
                'suffix': '-WGS84'
            },
            '5': {
                'func': gcj02_to_bd09,
                'source': '高德',
                'target': '百度',
                'suffix': '-百度'
            },
            '6': {
                'func': bd09_to_gcj02,
                'source': '百度',
                'target': '高德',
                'suffix': '-高德'
            }
        }
        
        params = conversion_params[mode]
        output_filename = f"{filename_without_ext}{params['suffix']}.xlsx"
        output_file = os.path.join(base_path, output_filename)
        
        # 记录日志
        logging.debug(f"转换模式: {params['source']}转{params['target']}")
        logging.debug(f"输入文件: {input_file}")
        logging.debug(f"输出文件: {output_file}")
        
        print(f"\n开始处理文件: {input_filename}")
        print(f"转换模式: {params['source']}转{params['target']}")
        
        # 执行转换
        batch_convert(input_file, output_file, params['func'], params['source'], params['target'])
        
        print(f"坐标转换完成！")
        print(f"结果已保存到: {output_filename}")
    except ValueError as e:
        # 处理列名找不到的错误
        error_msg = f"数据格式错误: {str(e)}"
        logging.error(error_msg)
        print(f"\n错误: {error_msg}")
        print("请确保Excel文件包含正确的经纬度列名。")
        print("程序会尝试识别包含特定关键词的列名，例如：")
        print("- WGS84坐标：包含'WGS84'、'84'等关键词的经度和纬度列")
        print("- 高德坐标：包含'高德'、'GCJ02'等关键词的经度和纬度列")
        print("- 百度坐标：包含'百度'、'BD09'等关键词的经度和纬度列")
    except Exception as e:
        # 处理其他错误
        error_msg = f"处理过程中出错: {str(e)}"
        logging.error(error_msg)
        print(f"\n错误: {error_msg}")
        print("请确保Excel文件格式正确，并且有读写权限。")
    
    # 等待用户按键退出
    input("\n按回车键退出...")


# 程序入口
if __name__ == "__main__":
    main()