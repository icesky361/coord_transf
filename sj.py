import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # 新增：用于保存和加载模型
import os  # 新增：用于文件路径操作

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class SJCoordinateConverter:
    """
    坐标转换模型类
    
    该类实现了基于机器学习的坐标转换，支持以下两种转换方向：
    1. 高德坐标系到思极坐标系
    2. 思极坐标系到高德坐标系
    通过已知的坐标对训练模型，然后使用模型预测新的坐标转换关系。
    支持随机森林和XGBoost两种算法。
    """
    def __init__(self):
        """初始化转换器实例"""
        self.model = None      # 转换模型
        self.X_train = None    # 训练特征数据
        self.X_test = None     # 测试特征数据
        self.y_train = None    # 训练目标数据
        self.y_test = None     # 测试目标数据
        self.y_pred = None     # 预测结果
        self.direction = None  # 转换方向: 'gaode_to_sj' 或 'sj_to_gaode'

    def load_data(self, file_path="思级高德互转_坐标数据源.xlsx"):
        """
        加载Excel格式的坐标数据
        
        Args:
            file_path (str): 数据文件路径，默认为"坐标数据.xlsx"
        
        Returns:
            pd.DataFrame: 加载的数据框
        
        Raises:
            ValueError: 当数据中缺少必要的列时
            Exception: 加载文件时的其他错误
        """
        try:
            df = pd.read_excel(file_path)
            print(f"成功加载数据，共{len(df)}条记录")
            # 检查必要的列是否存在
            required_columns = ["高德经度", "高德纬度", "思极经度", "思极纬度"]
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"数据中缺少必要的列: {col}")
            return df
        except Exception as e:
            print(f"加载数据时出错: {e}")
            raise

    def prepare_data(self, df, direction='gaode_to_sj', test_size=0.2, random_state=42):
        """
        准备训练数据和测试数据
        
        根据指定的转换方向，选择相应的特征和目标值
        
        Args:
            df (pd.DataFrame): 包含坐标数据的数据框
            direction (str): 转换方向，可选'gaode_to_sj'（高德到思极）或'sj_to_gaode'（思极到高德）
            test_size (float): 测试集比例，默认为0.2
            random_state (int): 随机种子，保证结果可复现
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test) 训练和测试数据
        
        Raises:
            ValueError: 当指定了不支持的转换方向时
        """
        self.direction = direction
        
        if direction == 'gaode_to_sj':
            # 特征：高德经纬度
            X = df[["高德经度", "高德纬度"]].values.astype(np.float64)
            # 目标：思极经纬度偏移量
            df["经度偏移"] = df["思极经度"] - df["高德经度"]
            df["纬度偏移"] = df["思极纬度"] - df["高德纬度"]
            y = df[["经度偏移", "纬度偏移"]].values.astype(np.float64)
        elif direction == 'sj_to_gaode':
            # 特征：思极经纬度
            X = df[["思极经度", "思极纬度"]].values.astype(np.float64)
            # 目标：高德经纬度偏移量
            df["经度偏移"] = df["高德经度"] - df["思极经度"]
            df["纬度偏移"] = df["高德纬度"] - df["思极纬度"]
            y = df[["经度偏移", "纬度偏移"]].values.astype(np.float64)
        else:
            raise ValueError(f"不支持的转换方向: {direction}，请使用'gaode_to_sj'或'sj_to_gaode'")

        # 划分训练集和测试集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        print(f"数据集划分完成: 训练集{len(self.X_train)}条, 测试集{len(self.X_test)}条")
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_model(self, model_type="xgboost"):
        """
        训练坐标转换模型
        
        Args:
            model_type (str): 模型类型，可选"random_forest"或"xgboost"，默认为"xgboost"
        
        Returns:
            MultiOutputRegressor: 训练好的模型
        
        Raises:
            ValueError: 当模型类型不支持或未准备数据时
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("请先调用prepare_data方法准备数据")

        print(f"开始训练{model_type}模型...")
        if model_type == "random_forest":
            # 使用随机森林
            model = MultiOutputRegressor(
                RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            )
        elif model_type == "xgboost":
            # 使用XGBoost
            model = MultiOutputRegressor(
                xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

        model.fit(self.X_train, self.y_train)
        print("模型训练完成")
        return model

    def save_model(self, model, model_path=None):
        """
        保存训练好的模型到model文件夹
        
        Args:
            model: 训练好的模型
            model_path (str, optional): 模型保存路径，如果为None则根据转换方向自动生成
        """
        # 确保model文件夹存在
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        if model_path is None:
            if self.direction == 'gaode_to_sj':
                model_path = os.path.join(model_dir, 'gaode_to_sj_model.pkl')
            elif self.direction == 'sj_to_gaode':
                model_path = os.path.join(model_dir, 'sj_to_gaode_model.pkl')
            else:
                model_path = os.path.join(model_dir, 'coordinate_model.pkl')
            
        joblib.dump(model, model_path)
        print(f"模型已保存到: {os.path.abspath(model_path)}")

    def load_model(self, model_path="coordinate_model.pkl"):
        """
        从文件加载训练好的模型
        
        Args:
            model_path (str): 模型文件路径，默认为"coordinate_model.pkl"
        
        Returns:
            加载的模型
        
        Raises:
            FileNotFoundError: 当模型文件不存在时
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        model = joblib.load(model_path)
        print(f"模型已从{os.path.abspath(model_path)}加载成功")
        return model

    def evaluate_model(self, model):
        """
        评估模型性能
        
        Args:
            model: 要评估的模型
        
        Returns:
            tuple: (mse, r2) 均方误差和决定系数
        
        Raises:
            ValueError: 当未准备测试数据时
        """
        if self.X_test is None or self.y_test is None:
            raise ValueError("请先调用prepare_data方法准备数据")

        # 预测
        self.y_pred = model.predict(self.X_test)

        # 计算MSE和R2
        mse = mean_squared_error(self.y_test, self.y_pred)
        r2 = r2_score(self.y_test, self.y_pred)
        print(f"模型评估结果:\n- 均方误差(MSE): {mse:.10f}\n- 决定系数(R2): {r2:.6f}")

        # 计算经纬度各自的误差
        lng_mse = mean_squared_error(self.y_test[:, 0], self.y_pred[:, 0])
        lat_mse = mean_squared_error(self.y_test[:, 1], self.y_pred[:, 1])
        print(f"经度偏移MSE: {lng_mse:.10f}")
        print(f"纬度偏移MSE: {lat_mse:.10f}")

        return mse, r2

    def visualize_errors(self):
        """
        可视化误差分布
        
        生成误差直方图和散点图，保存为"误差分析.png"
        
        Raises:
            ValueError: 当未进行模型评估时
        """
        if self.y_pred is None or self.y_test is None:
            raise ValueError("请先调用evaluate_model方法评估模型")

        # 计算误差
        errors = self.y_pred - self.y_test
        lng_errors = errors[:, 0] * 1e6  # 转换为微度
        lat_errors = errors[:, 1] * 1e6

        # 创建画布
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 经度误差直方图
        axes[0, 0].hist(lng_errors, bins=50, alpha=0.7, color='blue')
        axes[0, 0].set_title('经度偏移误差分布 (微度)')
        axes[0, 0].set_xlabel('误差 (微度)')
        axes[0, 0].set_ylabel('频率')
        axes[0, 0].axvline(x=0, color='r', linestyle='--')

        # 纬度误差直方图
        axes[0, 1].hist(lat_errors, bins=50, alpha=0.7, color='green')
        axes[0, 1].set_title('纬度偏移误差分布 (微度)')
        axes[0, 1].set_xlabel('误差 (微度)')
        axes[0, 1].set_ylabel('频率')
        axes[0, 1].axvline(x=0, color='r', linestyle='--')

        # 经度误差散点图
        axes[1, 0].scatter(self.y_test[:, 0] * 1e6, lng_errors, alpha=0.5, color='blue')
        axes[1, 0].set_title('经度偏移预测误差 vs 实际偏移')
        axes[1, 0].set_xlabel('实际偏移 (微度)')
        axes[1, 0].set_ylabel('预测误差 (微度)')
        axes[1, 0].axhline(y=0, color='r', linestyle='--')

        # 纬度误差散点图
        axes[1, 1].scatter(self.y_test[:, 1] * 1e6, lat_errors, alpha=0.5, color='green')
        axes[1, 1].set_title('纬度偏移预测误差 vs 实际偏移')
        axes[1, 1].set_xlabel('实际偏移 (微度)')
        axes[1, 1].set_ylabel('预测误差 (微度)')
        axes[1, 1].axhline(y=0, color='r', linestyle='--')

        plt.tight_layout()
        # 确保model文件夹存在
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # 保存误差分析图表到model文件夹
        error_plot_path = os.path.join(model_dir, '误差分析.png')
        plt.savefig(error_plot_path, dpi=300, bbox_inches='tight')
        print(f"误差可视化完成，已保存到: {os.path.abspath(error_plot_path)}")
        plt.show()

    def convert(self, model, input_lng, input_lat, direction=None):
        """
        坐标转换
        
        根据指定的转换方向，将输入坐标转换为目标坐标
        
        Args:
            model: 训练好的模型
            input_lng (float): 输入坐标系经度
            input_lat (float): 输入坐标系纬度
            direction (str, optional): 转换方向，如果为None则使用训练时的方向
        
        Returns:
            tuple: (target_lng, target_lat) 目标坐标系的经度和纬度
        
        Raises:
            ValueError: 当未指定转换方向且训练时也未设置时
        """
        # 确定转换方向
        convert_direction = direction or self.direction
        if convert_direction is None:
            raise ValueError("未指定转换方向，请提供direction参数或先调用prepare_data方法")

        # 确保输入是浮点数
        input_lng = float(input_lng)
        input_lat = float(input_lat)

        # 预测偏移量
        offset = model.predict(np.array([[input_lng, input_lat]]))[0]

        # 根据转换方向计算目标坐标
        if convert_direction == 'gaode_to_sj':
            target_lng = input_lng + offset[0]
            target_lat = input_lat + offset[1]
        elif convert_direction == 'sj_to_gaode':
            target_lng = input_lng + offset[0]
            target_lat = input_lat + offset[1]
        else:
            raise ValueError(f"不支持的转换方向: {convert_direction}")

        return target_lng, target_lat

    def test_conversion(self, model, sample_size=10):
        """
        测试转换功能
        
        随机选择测试集中的样本进行转换测试，并输出结果
        
        Args:
            model: 训练好的模型
            sample_size (int): 测试样本数量，默认为10
        
        Raises:
            ValueError: 当未准备测试数据时
        """
        if self.X_test is None:
            raise ValueError("请先调用prepare_data方法准备数据")

        # 随机选择样本
        indices = np.random.choice(len(self.X_test), min(sample_size, len(self.X_test)), replace=False)
        samples = self.X_test[indices]

        print("\n转换测试结果:")
        print("{:<12} {:<12} {:<12} {:<12} {:<12} {:<12}".format(
            "高德经度", "高德纬度", "预测思极经度", "预测思极纬度", "实际思极经度", "实际思极纬度"
        ))

        for i, (lng, lat) in enumerate(samples):
            sj_lng, sj_lat = self.convert(model, lng, lat)
            actual_sj_lng = lng + self.y_test[indices[i], 0]
            actual_sj_lat = lat + self.y_test[indices[i], 1]
            print("{:<12.8f} {:<12.8f} {:<12.8f} {:<12.8f} {:<12.8f} {:<12.8f}".format(
                lng, lat, sj_lng, sj_lat, actual_sj_lng, actual_sj_lat
            ))

if __name__ == "__main__":
    # 创建转换器实例
    converter = SJCoordinateConverter()

    try:
        # 加载数据
        df = converter.load_data()

        # 询问用户想要训练的转换方向
        print("请选择要训练的转换方向：")
        print("1. 高德坐标到思极坐标 (gaode_to_sj)")
        print("2. 思极坐标到高德坐标 (sj_to_gaode)")
        choice = input("请输入选择 (1/2): ")
        
        if choice == '1':
            direction = 'gaode_to_sj'
        elif choice == '2':
            direction = 'sj_to_gaode'
        else:
            print("无效的选择，默认使用高德坐标到思极坐标")
            direction = 'gaode_to_sj'

        # 准备数据
        converter.prepare_data(df, direction=direction)

        # 训练模型 - 可以尝试"random_forest"或"xgboost"
        model = converter.train_model(model_type="xgboost")

        # 保存模型
        converter.save_model(model)

        # 评估模型
        converter.evaluate_model(model)

        # 可视化误差
        converter.visualize_errors()

        # 测试转换
        converter.test_conversion(model)

        print(f"\n{direction}转换模型已成功训练和测试。")
        if direction == 'gaode_to_sj':
            print("您可以使用converter.convert(model, gaode_lng, gaode_lat)函数进行坐标转换。")
        else:
            print("您可以使用converter.convert(model, sj_lng, sj_lat)函数进行坐标转换。")
        print(f"模型已保存到model文件夹下的'{direction}_model.pkl'文件。")

    except Exception as e:
        print(f"程序执行出错: {e}")