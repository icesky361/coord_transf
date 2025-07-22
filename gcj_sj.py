import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from pyproj import Transformer
from sklearn.linear_model import Ridge
import xgboost as xgb  # 添加此行导入xgboost

# 1. 加载数据（示例格式）
# 假设CSV包含字段: gd_lon, gd_lat, sj_lon, sj_lat
data = pd.read_excel("坐标数据.xlsx", sheet_name="坐标数据")  # 改为xlsx读取
# 在读取数据后添加严格清洗
import re

def clean_coord(value):
    if pd.isna(value):
        return np.nan
    try:
        # 移除所有非数字和负号、小数点字符
        cleaned = re.sub(r'[^0-9\-\.]', '', str(value))
        return float(cleaned)
    except:
        return np.nan

# 应用清洗函数
for col in ['高德经度','高德纬度','思极经度','思极纬度']:
    data[col] = data[col].apply(clean_coord)

# 再次检查
print("清洗后数据统计:")
print(data.describe())

# 定义坐标变量
gd_coords = data[['高德经度', '高德纬度']].values
sj_coords = data[['思极经度', '思极纬度']].values

# 数据标准化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
gd_coords_scaled = scaler.fit_transform(gd_coords)
poly = PolynomialFeatures(degree=1, include_bias=False)
gd_poly = poly.fit_transform(gd_coords)
# 检查数值范围
max_val = np.max(np.abs(gd_coords))
if max_val > 1e6:
    print(f'警告：发现极大值 {max_val}，可能影响数值稳定性')
print("多项式特征值范围:", np.min(gd_poly), np.max(gd_poly))

print("非数字值检查:")
print(data.isna().sum())


# 删除包含NaN的行
data = data.dropna()

# 多项式拟合前添加数据缩放
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_coords = scaler.fit_transform(gd_coords)
# 在读取数据后添加检查
print(data.describe())  # 查看数据统计信息
print(data.isnull().sum())  # 检查空值
# 添加数据验证
print("数据统计:")
print(data.describe())
print("缺失值检查:")
print(data.isnull().sum())
print("无限值检查:")
print(np.isinf(data[['高德经度', '高德纬度', '思极经度', '思极纬度']].values).sum())
# 在多项式拟合前添加
if not all((-180 <= data[['高德经度','思极经度']]).all() <= 180) or \
   not all((-90 <= data[['高德纬度','思极纬度']]).all() <= 90):
    print("警告：存在超出正常范围的坐标值")
    data = data[(-180 <= data[['高德经度','思极经度']]) & 
               (data[['高德经度','思极经度']] <= 180) &
               (-90 <= data[['高德纬度','思极纬度']]) &
               (data[['高德纬度','思极纬度']] <= 90)]
# 替换无穷大值为NaN
import numpy as np
data = data.replace([np.inf, -np.inf], np.nan)
# 删除包含NaN的行
data = data.dropna()
# 添加这行：在数据清洗后重新定义坐标数组
gd_coords = data[['gd_x', 'gd_y']].values
sj_coords = data[['sj_x', 'sj_y']].values
# 2. 计算平面直角坐标（避免球面距离误差）[2](@ref)
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857")  # WGS84转Web墨卡托
data["gd_x"], data["gd_y"] = transformer.transform(data["高德经度"], data["高德纬度"])
data["sj_x"], data["sj_y"] = transformer.transform(data["思极经度"], data["思极纬度"])

# 3. 构建多项式特征（增强非线性拟合能力）[7,8](@ref)
# 多项式特征生成（降为1次）

gd_poly = poly.fit_transform(gd_coords)

# 添加Ridge正则化模型
model = Ridge(alpha=1.0)
model.fit(gd_poly, sj_coords)

poly_features = poly.get_feature_names_out(["gd_x", "gd_y"])
X = pd.DataFrame(gd_poly, columns=poly_features)

# 4. 定义目标变量：思极坐标偏移量
y_x = data["sj_x"] - data["gd_x"]
y_y = data["sj_y"] - data["gd_y"]

# 添加标签数据清洗
mask = ~np.isnan(y_x) & ~np.isnan(y_y) & ~np.isinf(y_x) & ~np.isinf(y_y)
X = X[mask]
y_x = y_x[mask]
y_y = y_y[mask]

# 添加数据检查确保样本不为空
if len(X) == 0:
    raise ValueError("掩码操作后样本数量为0，请检查数据清洗逻辑")
print(f"掩码后剩余样本数量: {len(X)}")

# 数据拆分步骤 - 修复参数顺序
from sklearn.model_selection import train_test_split
# 使用元组形式正确拆分多目标变量
(X_train, X_test), (yx_train, yx_test), (yy_train, yy_test) = train_test_split(
    X, y_x, y_y, test_size=0.2, random_state=42
)

# 经度偏移模型
dtrain_x = xgb.DMatrix(X_train, label=yx_train)
params_x = {
    'objective': 'reg:squarederror',
    'max_depth': 8,
    'eta': 0.1,
    'subsample': 0.8,
    'lambda': 1.5
}
xgb_x = xgb.train(params_x, dtrain_x, num_boost_round=300)

# 纬度偏移模型
dtrain_y = xgb.DMatrix(X_train, label=yy_train)
params_y = {
    'objective': 'reg:squarederror',
    'max_depth': 8,
    'eta': 0.1,
    'subsample': 0.8,
    'lambda': 1.5
}
xgb_y = xgb.train(params_y, dtrain_y, num_boost_round=300)

from sklearn.metrics import mean_squared_error


# XGBoost评估
dtest = xgb.DMatrix(X_test)
xgb_pred_x = xgb_x.predict(dtest)
xgb_pred_y = xgb_y.predict(dtest)
xgb_mse_x = mean_squared_error(yx_test, xgb_pred_x)
xgb_mse_y = mean_squared_error(yy_test, xgb_pred_y)

print(f"XGBoost MSE - X: {xgb_mse_x:.4f}, Y: {xgb_mse_y:.4f}")


def convert_coord(lon, lat):
    """高德坐标转思极坐标"""
    # 坐标转换
    x, y = transformer.transform(lat, lon)
    
    # 生成多项式特征
    point = pd.DataFrame([[x, y]], columns=["gd_x", "gd_y"])
    point_poly = poly.transform(point)
    
    # 预测偏移量
    delta_x = 0.3 * rf_x.predict(point_poly)[0] + 0.7 * xgb_x.predict(xgb.DMatrix(point_poly))
    delta_y = 0.3 * rf_y.predict(point_poly)[0] + 0.7 * xgb_y.predict(xgb.DMatrix(point_poly))
    
    # 计算思极坐标
    sj_x = x + delta_x
    sj_y = y + delta_y
    
    # 转回经纬度
    sj_lat, sj_lon = transformer.transform(sj_x, sj_y, direction="INVERSE")
    return sj_lon, sj_lat

# 坐标转换后检查
print("转换后坐标NaN检查:")
print(data[["gd_x", "gd_y", "sj_x", "sj_y"]].isna().sum())

data = data.dropna(subset=["gd_x", "gd_y", "sj_x", "sj_y"])
