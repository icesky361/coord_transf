import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class SJCoordinateConverter:
    def __init__(self):
        self.model_lng = None
        self.model_lat = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None

    def load_data(self, file_path="坐标数据.xlsx"):
        """加载Excel数据"""
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

    def prepare_data(self, df, test_size=0.2, random_state=42):
        """准备训练数据和测试数据"""
        # 特征：高德经纬度
        X = df[["高德经度", "高德纬度"]].values.astype(np.float64)
        # 目标：思极经纬度偏移量（而不是直接预测思极坐标）
        df["经度偏移"] = df["思极经度"] - df["高德经度"]
        df["纬度偏移"] = df["思极纬度"] - df["高德纬度"]
        y = df[["经度偏移", "纬度偏移"]].values.astype(np.float64)

        # 划分训练集和测试集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        print(f"数据集划分完成: 训练集{len(self.X_train)}条, 测试集{len(self.X_test)}条")
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_model(self, model_type="xgboost"):
        """训练模型"""
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

    def evaluate_model(self, model):
        """评估模型性能"""
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
        """可视化误差分布"""
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
        plt.savefig('误差分析.png', dpi=300, bbox_inches='tight')
        print("误差可视化完成，已保存为'误差分析.png'")
        plt.show()

    def convert(self, model, gaode_lng, gaode_lat):
        """将高德坐标转换为思极坐标"""
        # 确保输入是浮点数
        gaode_lng = float(gaode_lng)
        gaode_lat = float(gaode_lat)

        # 预测偏移量
        offset = model.predict(np.array([[gaode_lng, gaode_lat]]))[0]

        # 计算思极坐标
        sj_lng = gaode_lng + offset[0]
        sj_lat = gaode_lat + offset[1]

        return sj_lng, sj_lat

    def test_conversion(self, model, sample_size=10):
        """测试转换功能"""
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

        # 准备数据
        converter.prepare_data(df)

        # 训练模型 - 可以尝试"random_forest"或"xgboost"
        model = converter.train_model(model_type="xgboost")

        # 评估模型
        converter.evaluate_model(model)

        # 可视化误差
        converter.visualize_errors()

        # 测试转换
        converter.test_conversion(model)

        print("\n坐标转换模型已成功训练和测试。")
        print("您可以使用converter.convert(model, gaode_lng, gaode_lat)函数进行坐标转换。")

    except Exception as e:
        print(f"程序执行出错: {e}")