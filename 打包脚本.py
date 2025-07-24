import os
import sys
import subprocess

# 指定虚拟环境路径
VENV_PATH = r"D:\soft\anaconda\envs\coord_env"
# 指定要打包的主程序
MAIN_SCRIPT = r"d:\program\python\PythonProject\经纬度坐标转化\综合坐标转换工具.py"
# 输出目录
OUTPUT_DIR = r"d:\program\python\PythonProject\经纬度坐标转化\release-binaries"

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 构建xgboost.dll路径
xgboost_dll_path = os.path.join(VENV_PATH, "Lib", "site-packages", "xgboost", "lib", "xgboost.dll")

# 构建PyInstaller命令
pyinstaller_cmd = [
    # 使用虚拟环境中的Python解释器
    os.path.join(VENV_PATH, "python.exe"),
    # 调用虚拟环境中的PyInstaller
    os.path.join(VENV_PATH, "Scripts", "pyinstaller.exe"),
    # 单文件模式
    "-F",
    # 不显示控制台窗口（如果需要GUI界面）
    # "-w",
    # 添加模型文件夹
    "--add-data", r"d:\program\python\PythonProject\经纬度坐标转化\model;model",
    # 隐藏导入的库
    "--hidden-import", "sklearn",
    "--hidden-import", "sklearn.multioutput",
    "--hidden-import", "xgboost",
    # 添加xgboost.dll文件
    "--add-binary", f"{xgboost_dll_path};.",
    "--add-binary", f"{xgboost_dll_path};xgboost/lib",
    # 添加xgboost VERSION文件
    "--add-data", os.path.join(VENV_PATH, "Lib", "site-packages", "xgboost", "VERSION") + ";xgboost",
    # 输出目录
    "--distpath", OUTPUT_DIR,
    # 主程序
    MAIN_SCRIPT
]

print("开始打包...")
print(f"命令: {' '.join(pyinstaller_cmd)}")

# 执行打包命令
try:
    result = subprocess.run(
        pyinstaller_cmd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    print("打包成功!")
    print(f"输出日志:\n{result.stdout}")
    print(f"可执行文件已生成在: {OUTPUT_DIR}")
except subprocess.CalledProcessError as e:
    print(f"打包失败: {e}")
    print(f"错误输出:\n{e.stderr}")
    sys.exit(1)