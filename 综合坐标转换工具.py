import os
import sys
import pandas as pd
import math
import joblib
import numpy as np
import matplotlib.pyplot as plt
# 确保中文显示正常

plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

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

# 思极坐标转换类
class CoordinateInferenceConverter:
    """
    坐标转换推理工具类
    
    该类实现了加载已训练的模型，并支持两种方向的坐标转换：
    1. 高德坐标到思极坐标
    2. 思极坐标到高德坐标
    支持从Excel文件读取数据并将转换结果保存到新的Excel文件中。
    """
    def __init__(self):
        """初始化推理转换器实例"""
        self.model = None  # 存储加载的模型
        self.direction = None  # 转换方向
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

    def load_model(self, direction):
        """
        根据转换方向加载相应的已训练模型
        
        Args:
            direction (str): 转换方向，'gaode_to_sj'或'sj_to_gaode'
        
        Returns:
            bool: 加载成功返回True，否则返回False
        """
        self.direction = direction
        
        # 从model文件夹加载模型
        if direction == 'gaode_to_sj':
            model_path = os.path.join(self.base_path, 'model', 'gaode_to_sj_model.pkl')
        elif direction == 'sj_to_gaode':
            model_path = os.path.join(self.base_path, 'model', 'sj_to_gaode_model.pkl')
        else:
            print(f"错误: 不支持的转换方向 - {direction}")
            return False

        try:
            if not os.path.exists(model_path):
                print(f"错误: 模型文件不存在 - {model_path}")
                print("请先使用sj.py脚本训练相应的模型")
                return False

            self.model = joblib.load(model_path)
            print(f"成功加载{direction}模型: {model_path}")
            return True
        except Exception as e:
            print(f"加载模型时出错: {str(e)}")
            return False

    def find_coordinate_columns(self, df):
        """
        根据转换方向智能识别相应的坐标列
        
        Args:
            df (pd.DataFrame): 包含坐标数据的数据框
        
        Returns:
            tuple: (lng_col, lat_col) 经度列名和纬度列名，如果未找到则返回(None, None)
        """
        if self.direction == 'gaode_to_sj':
            # 定义高德坐标的关键词
            keywords = ['高德', 'gcj02', 'gcj', '02']
            column_type = '高德'
        elif self.direction == 'sj_to_gaode':
            # 定义思极坐标的关键词
            keywords = ['思极', 'sj']
            column_type = '思极'
        else:
            print("错误: 未设置转换方向")
            return None, None

        lng_keywords = ['经度', 'longitude', 'lng']
        lat_keywords = ['纬度', 'latitude', 'lat']

        # 使用通用函数查找经度列
        lng_col = find_column_by_keywords(df, keywords, '经度')
        # 使用通用函数查找纬度列
        lat_col = find_column_by_keywords(df, keywords, '纬度')

        if lng_col and lat_col:
            print(f"识别到{column_type}坐标列: 经度='{lng_col}', 纬度='{lat_col}'")
        else:
            print(f"错误: 无法识别{column_type}经纬度列")

        return lng_col, lat_col

    def convert_coordinates(self, df, lng_col, lat_col):
        """
        批量转换坐标
        
        Args:
            df (pd.DataFrame): 包含坐标数据的数据框
            lng_col (str): 经度列名
            lat_col (str): 纬度列名
        
        Returns:
            pd.DataFrame: 包含转换结果的数据框，添加了目标坐标列
        """
        # 提取输入坐标
        input_coords = df[[lng_col, lat_col]].values.astype(np.float64)

        # 预测偏移量
        offsets = self.model.predict(input_coords)

        # 计算目标坐标
        target_lng = input_coords[:, 0] + offsets[:, 0]
        target_lat = input_coords[:, 1] + offsets[:, 1]

        # 添加到数据框
        if self.direction == 'gaode_to_sj':
            df['思极经度'] = target_lng
            df['思极纬度'] = target_lat
        else:
            df['高德经度'] = target_lng
            df['高德纬度'] = target_lat

        return df

# 中国地区经纬度合理范围常量
CHINA_LNG_MIN, CHINA_LNG_MAX = 73.66, 135.05
CHINA_LAT_MIN, CHINA_LAT_MAX = 3.86, 53.55

# 坐标范围校验函数
def check_coordinates_in_range(df, lng_col, lat_col):
    """
    检查坐标是否在中国地区合理范围内
    :param df: 数据框
    :param lng_col: 经度列名
    :param lat_col: 纬度列名
    :return: 超出范围的行信息列表（格式：[(DataFrame索引, Excel行号, 经度值, 纬度值)]）
    """
    lng_min, lng_max = CHINA_LNG_MIN, CHINA_LNG_MAX
    lat_min, lat_max = CHINA_LAT_MIN, CHINA_LAT_MAX
    out_of_range = []
    for idx, row in df.iterrows():
        try:
            lng = float(row[lng_col])
            lat = float(row[lat_col])
            if not (CHINA_LNG_MIN <= lng <= CHINA_LNG_MAX and CHINA_LAT_MIN <= lat <= CHINA_LAT_MAX):
                out_of_range.append((idx, idx + 2, lng, lat))  # Excel行号=DataFrame索引+2（首行是标题）
        except ValueError:
            out_of_range.append((idx, idx + 2, row[lng_col], row[lat_col]))
    return out_of_range

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
def batch_convert(input_file, output_file, convert_func, source_type, target_type, converter=None):
    """
    批量转换坐标
    :param input_file: 输入文件路径
    :param output_file: 输出文件路径
    :param convert_func: 转换函数
    :param source_type: 源坐标系类型，如'WGS84'、'高德'、'百度'或'思极'
    :param target_type: 目标坐标系类型，如'WGS84'、'高德'、'百度'或'思极'
    :param converter: 思极坐标转换器实例，仅在涉及思极坐标转换时使用
    """
    # 读取输入文件
    df = pd.read_excel(input_file, engine='openpyxl')
    
    # 定义坐标系关键词映射
    keywords_map = {
        'WGS84': ['WGS84', '84', 'wgs'],
        '高德': ['高德', 'GCJ02', 'gcj', '02'],
        '百度': ['百度', 'BD09', 'bd', '09'],
        '思极': ['思极', 'sj']
    }
    
    # 查找源坐标系的经纬度列
    source_lng_col = find_column_by_keywords(df, keywords_map[source_type], '经度')
    source_lat_col = find_column_by_keywords(df, keywords_map[source_type], '纬度')
    
    if source_lng_col is None or source_lat_col is None:
        raise ValueError(f"无法找到{source_type}经纬度列，请确保Excel文件包含经度和纬度列")
    
    # 提取坐标范围校验函数
    out_of_range = check_coordinates_in_range(df, source_lng_col, source_lat_col)
    
    # 创建超出范围的行索引集合
    out_of_range_indices = {idx for idx, _, _, _ in out_of_range}
    
    if out_of_range:
        print(f"\n警告：发现以下坐标超出中国地区经纬度范围（经度: {CHINA_LNG_MIN}°~{CHINA_LNG_MAX}°，纬度: {CHINA_LAT_MIN}°~{CHINA_LAT_MAX}°）：")
        print("行号\t经度\t纬度")
        for _, row_num, lng, lat in out_of_range:
            print(f"{row_num}\t{lng}\t{lat}")
        
        # 使用通用交互函数确认继续
        if not confirm_continue("\n是否继续转换？(y/n): "):
            print("程序已终止，未执行坐标转换。")
            return
        else:
            print("将继续转换，超出范围的坐标将标记为'原坐标不在合理范围'。")
    else:
        print("\n所有坐标均在合理范围内。")
    
    # 执行坐标转换
    if converter is not None:
        # 使用思极转换器，并处理超出范围的坐标
        # 确定思极转换生成的列名
        if target_type == '思极':
            target_lng_col = '思极经度'
            target_lat_col = '思极纬度'
        else:
            target_lng_col = '高德经度'
            target_lat_col = '高德纬度'
        
        if out_of_range_indices:
            # 先创建一个副本，避免修改原始数据
            temp_df = df.copy()
            # 对所有行执行转换
            temp_df = converter.convert_coordinates(temp_df, source_lng_col, source_lat_col)
            
            # 对超出范围的行进行标记
            for idx in out_of_range_indices:
                if idx < len(df):
                    df.at[idx, target_lng_col] = '原坐标不在合理范围'
                    df.at[idx, target_lat_col] = '原坐标不在合理范围'
            
            # 对范围内的行复制转换结果
            for idx in range(len(df)):
                if idx not in out_of_range_indices:
                    df.at[idx, target_lng_col] = temp_df.at[idx, target_lng_col]
                    df.at[idx, target_lat_col] = temp_df.at[idx, target_lat_col]
        else:
            # 所有坐标都在范围内，直接转换
            df = converter.convert_coordinates(df, source_lng_col, source_lat_col)

    else:
        # 输出列名
        target_lng_col = f'经度（{target_type}计算）'
        target_lat_col = f'纬度（{target_type}计算）'
        # 使用普通转换函数，并处理超出范围的坐标
        if out_of_range_indices:
            # 创建目标列并初始化为'原坐标不在合理范围'
            df[target_lng_col] = '原坐标不在合理范围'
            df[target_lat_col] = '原坐标不在合理范围'
            
            # 只对范围内的行应用转换函数
            for idx in range(len(df)):
                if idx not in out_of_range_indices:
                    try:
                        lng = float(df.at[idx, source_lng_col])
                        lat = float(df.at[idx, source_lat_col])
                        if china_lng_min <= lng <= china_lng_max and china_lat_min <= lat <= china_lat_max:
                            converted_lng, converted_lat = convert_func(lng, lat)
                            df.at[idx, target_lng_col] = converted_lng
                            df.at[idx, target_lat_col] = converted_lat
                    except ValueError:
                        pass  # 保持标记不变
        else:
            # 所有坐标都在范围内，直接转换
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
    
    # 确保输出目录存在
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
            print("2. WGS84 → 百度(BD-09)")
            print("3. WGS84 → 思极")
            print("4. 思极 → 高德(GCJ-02)")
            print("5. 思极 → 百度(BD-09)")
            print("6. 思极 → WGS84")
            print("7. 高德(GCJ-02) → WGS84")
            print("8. 高德(GCJ-02) → 百度(BD-09)")
            print("9. 高德(GCJ-02) → 思极")
            print("10. 百度(BD-09) → WGS84")
            print("11. 百度(BD-09) → 高德(GCJ-02)")
            print("12. 百度(BD-09) → 思极")
            mode = input("请选择(1-12): ")
            if mode in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']:
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
        
        # 提示用户输入文件序号
        while True:
            try:
                file_index = int(input("\n请输入文件序号(1-{}): ".format(len(excel_files))))
                if 1 <= file_index <= len(excel_files):
                    input_filename = excel_files[file_index - 1]
                    input_file = os.path.join(base_path, input_filename)
                    break
                else:
                    print(f"输入无效，请输入1到{len(excel_files)}之间的数字！")
            except ValueError:
                print("输入无效，请输入一个数字！")
        
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
                'func': wgs84_to_bd09,
                'source': 'WGS84',
                'target': '百度',
                'suffix': '-百度'
            },
            '3': {
                'func': None,
                'source': 'WGS84',
                'target': '思极',
                'suffix': '-思极',
                'sj_direction': 'gaode_to_sj'  # WGS84→思极需要先转高德再转思极
            },
            '4': {
                'func': None,
                'source': '思极',
                'target': '高德',
                'suffix': '-高德',
                'sj_direction': 'sj_to_gaode'
            },
            '5': {
                'func': None,
                'source': '思极',
                'target': '百度',
                'suffix': '-百度',
                'sj_direction': 'sj_to_gaode'  # 思极→百度需要先转高德再转百度
            },
            '6': {
                'func': None,
                'source': '思极',
                'target': 'WGS84',
                'suffix': '-WGS84',
                'sj_direction': 'sj_to_gaode'  # 思极→WGS84需要先转高德再转WGS84
            },
            '7': {
                'func': gcj02_to_wgs84,
                'source': '高德',
                'target': 'WGS84',
                'suffix': '-WGS84'
            },
            '8': {
                'func': gcj02_to_bd09,
                'source': '高德',
                'target': '百度',
                'suffix': '-百度'
            },
            '9': {
                'func': None,
                'source': '高德',
                'target': '思极',
                'suffix': '-思极',
                'sj_direction': 'gaode_to_sj'
            },
            '10': {
                'func': bd09_to_wgs84,
                'source': '百度',
                'target': 'WGS84',
                'suffix': '-WGS84'
            },
            '11': {
                'func': bd09_to_gcj02,
                'source': '百度',
                'target': '高德',
                'suffix': '-高德'
            },
            '12': {
                'func': None,
                'source': '百度',
                'target': '思极',
                'suffix': '-思极',
                'sj_direction': 'gaode_to_sj'  # 百度→思极需要先转高德再转思极
            }
        }
        
        params = conversion_params[mode]
        output_filename = f"{filename_without_ext}{params['suffix']}.xlsx"
        output_file = os.path.join(base_path, 'run', output_filename)  # 保存到run文件夹
        
        # 记录日志
        logging.debug(f"转换模式: {params['source']}转{params['target']}")
        logging.debug(f"输入文件: {input_file}")
        logging.debug(f"输出文件: {output_file}")
        
        print(f"\n开始处理文件: {input_filename}")
        print(f"转换模式: {params['source']}转{params['target']}")
        
        # 执行转换
        converter = None
        if params.get('sj_direction'):
            # 涉及思极坐标转换
            converter = CoordinateInferenceConverter()
            # 对于需要多步转换的情况
            if mode == '3':  # WGS84 → 思极: 需要先转高德再转思极
                print("WGS84转思极需要先将WGS84转为高德，再将高德转为思极")
                # 先进行WGS84到高德的转换
                temp_df = pd.read_excel(input_file, engine='openpyxl')
                temp_df[['高德经度（第一步辅助转换）', '高德纬度（第一步辅助转换）']] = temp_df.apply(
                    lambda row: wgs84_to_gcj02(row[find_column_by_keywords(temp_df, ['WGS84', '84', 'wgs'], '经度')], 
                                             row[find_column_by_keywords(temp_df, ['WGS84', '84', 'wgs'], '纬度')]),
                    axis=1,
                    result_type='expand'
                )
                # 保存临时结果
                temp_file = os.path.join(base_path, 'run', f"{filename_without_ext}-temp.xlsx")
                temp_df.to_excel(temp_file, index=False, engine='openpyxl')
                # 加载高德到思极的模型
                if not converter.load_model('gaode_to_sj'):
                    print("无法加载模型，程序退出")
                    return
                # 使用临时文件作为输入
                input_file = temp_file
            elif mode == '5':  # 思极 → 百度: 需要先转高德再转百度
                print("思极转百度需要先将思极转为高德，再将高德转为百度")
                # 加载思极到高德的模型
                if not converter.load_model('sj_to_gaode'):
                    print("无法加载模型，程序退出")
                    return
                # 先进行思极到高德的转换
                temp_df = pd.read_excel(input_file, engine='openpyxl')
                # 识别思极坐标列
                sj_lng_col, sj_lat_col = find_column_by_keywords(temp_df, ['思极', 'sj'], '经度'), find_column_by_keywords(temp_df, ['思极', 'sj'], '纬度')
                if sj_lng_col is None or sj_lat_col is None:
                    raise ValueError("无法找到思极经纬度列")
                # 提取输入坐标
                input_coords = temp_df[[sj_lng_col, sj_lat_col]].values.astype(np.float64)
                # 预测偏移量
                offsets = converter.model.predict(input_coords)
                # 计算目标坐标
                target_lng = input_coords[:, 0] + offsets[:, 0]
                target_lat = input_coords[:, 1] + offsets[:, 1]
                # 添加到数据框
                temp_df['高德经度（第一步辅助转换）'] = target_lng
                temp_df['高德纬度（第一步辅助转换）'] = target_lat
                # 保存临时结果
                temp_file = os.path.join(base_path, 'run', f"{filename_without_ext}-temp.xlsx")
                temp_df.to_excel(temp_file, index=False, engine='openpyxl')
                # 使用临时文件作为输入
                input_file = temp_file
                # 执行高德到百度的转换
                batch_convert(input_file, output_file, gcj02_to_bd09, '高德', '百度')
                # 删除临时文件
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                print(f"坐标转换完成！")
                print(f"结果已保存到: {output_filename}")
                input("\n按回车键退出...")
                return
            elif mode == '6':  # 思极 → WGS84: 需要先转高德再转WGS84
                print("思极转WGS84需要先将思极转为高德，再将高德转为WGS84")
                # 加载思极到高德的模型
                if not converter.load_model('sj_to_gaode'):
                    print("无法加载模型，程序退出")
                    return
                # 先进行思极到高德的转换
                temp_df = pd.read_excel(input_file, engine='openpyxl')
                # 识别思极坐标列
                sj_lng_col, sj_lat_col = find_column_by_keywords(temp_df, ['思极', 'sj'], '经度'), find_column_by_keywords(temp_df, ['思极', 'sj'], '纬度')
                if sj_lng_col is None or sj_lat_col is None:
                    raise ValueError("无法找到思极经纬度列")
                # 提取输入坐标
                input_coords = temp_df[[sj_lng_col, sj_lat_col]].values.astype(np.float64)
                # 预测偏移量
                offsets = converter.model.predict(input_coords)
                # 计算目标坐标
                target_lng = input_coords[:, 0] + offsets[:, 0]
                target_lat = input_coords[:, 1] + offsets[:, 1]
                # 添加到数据框
                temp_df['高德经度（第一步辅助转换）'] = target_lng
                temp_df['高德纬度（第一步辅助转换）'] = target_lat
                # 保存临时结果
                temp_file = os.path.join(base_path, 'run', f"{filename_without_ext}-temp.xlsx")
                temp_df.to_excel(temp_file, index=False, engine='openpyxl')
                # 使用临时文件作为输入
                input_file = temp_file
                # 执行高德到WGS84的转换
                batch_convert(input_file, output_file, gcj02_to_wgs84, '高德', 'WGS84')
                # 删除临时文件
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                print(f"坐标转换完成！")
                print(f"结果已保存到: {output_filename}")
                input("\n按回车键退出...")
                return
            elif mode == '12':  # 百度 → 思极: 需要先转高德再转思极
                print("百度转思极需要先将百度转为高德，再将高德转为思极")
                # 先进行百度到高德的转换
                temp_df = pd.read_excel(input_file, engine='openpyxl')
                temp_df[['高德经度（第一步辅助转换）', '高德纬度（第一步辅助转换）']] = temp_df.apply(
                    lambda row: bd09_to_gcj02(row[find_column_by_keywords(temp_df, ['百度', 'BD09', 'bd', '09'], '经度')], 
                                             row[find_column_by_keywords(temp_df, ['百度', 'BD09', 'bd', '09'], '纬度')]),
                    axis=1,
                    result_type='expand'
                )
                # 保存临时结果
                temp_file = os.path.join(base_path, 'run', f"{filename_without_ext}-temp.xlsx")
                temp_df.to_excel(temp_file, index=False, engine='openpyxl')
                # 加载高德到思极的模型
                if not converter.load_model('gaode_to_sj'):
                    print("无法加载模型，程序退出")
                    return
                # 使用临时文件作为输入
                input_file = temp_file
            else:
                # 直接加载对应的模型
                if not converter.load_model(params['sj_direction']):
                    print("无法加载模型，程序退出")
                    return
            # 执行思极相关转换
            batch_convert(input_file, output_file, None, params['source'], params['target'], converter)
            # 如果有临时文件，删除它
            if 'temp_file' in locals() and os.path.exists(temp_file):
                os.remove(temp_file)
        else:
            # 执行普通转换
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
        print("- 思极坐标：包含'思极'、'sj'等关键词的经度和纬度列")
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