@echo off
REM ============================================================================
REM 人脸检测与去重流水线 - Windows 快速开始脚本
REM ============================================================================

setlocal enabledelayedexpansion

echo.
echo ========================================================================
echo 人脸检测与去重流水线 - Windows 安装向导
echo ========================================================================
echo.

REM 检查Python是否已安装
python --version > nul 2>&1
if errorlevel 1 (
    echo.
    echo ❌ 错误：未找到Python
    echo 请先从 https://www.python.org/downloads/ 下载安装Python
    echo 安装时请勾选 "Add Python to PATH"
    pause
    exit /b 1
)

echo ✅ 找到Python
python --version

REM 创建虚拟环境
if not exist "venv" (
    echo.
    echo 创建Python虚拟环境...
    python -m venv venv
)

REM 激活虚拟环境
echo.
echo 激活虚拟环境...
call venv\Scripts\activate.bat

REM 升级pip
echo.
echo 升级pip...
python -m pip install --upgrade pip

REM ============================================================================
REM 安装PyTorch
REM ============================================================================

echo.
echo ========================================================================
echo 安装PyTorch
echo ========================================================================
echo.
echo 请根据你的系统选择合适的安装命令：
echo.
echo [1] CUDA 12.x 用户 ^(推荐^)
echo     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
echo.
echo [2] CUDA 11.8 用户
echo     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
echo.
echo [3] CPU only ^(无CUDA^)
echo     pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
echo.
echo [4] Apple Silicon ^(M1/M2/M3^)
echo     pip install torch torchvision
echo.

set /p choice="请选择 [1-4，回车默认1]: "
if "!choice!"=="" set choice=1

if "!choice!"=="1" (
    echo.
    echo 安装 PyTorch CUDA 12.x ...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
) else if "!choice!"=="2" (
    echo.
    echo 安装 PyTorch CUDA 11.8 ...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
) else if "!choice!"=="3" (
    echo.
    echo 安装 PyTorch CPU ...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
) else if "!choice!"=="4" (
    echo.
    echo 安装 PyTorch Apple Silicon ...
    pip install torch torchvision
) else (
    echo 无效的选择，使用CUDA 12.x
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
)

REM ============================================================================
REM 安装其他依赖
REM ============================================================================

echo.
echo ========================================================================
echo 安装其他依赖包...
echo ========================================================================
pip install -r requirements.txt

REM ============================================================================
REM 下载模型
REM ============================================================================

echo.
echo ========================================================================
echo 下载预训练模型...
echo ========================================================================
echo.
echo 这将下载 InsightFace 和 YOLOv11-face 模型（约375MB）
echo 首次下载可能需要几分钟，请耐心等待...
echo.

python download_models.py

if errorlevel 1 (
    echo.
    echo ⚠️  模型下载失败，但不影响继续使用
    echo 系统会在首次运行时尝试自动下载
)

REM ============================================================================
REM 完成
REM ============================================================================

echo.
echo ========================================================================
echo ✅ 安装完成！
echo ========================================================================
echo.
echo 现在你可以运行以下命令之一来开始处理视频：
echo.
echo 基本使用（处理单个视频）：
echo   python face_dedup_pipeline.py input_video.mp4
echo.
echo 指定输出目录：
echo   python face_dedup_pipeline.py input_video.mp4 -o ./output
echo.
echo 使用GPU加速：
echo   python face_dedup_pipeline.py input_video.mp4 --cuda
echo.
echo 处理视频目录：
echo   python face_dedup_pipeline.py ./videos/ -o ./output
echo.
echo 查看所有可用参数：
echo   python face_dedup_pipeline.py --help
echo.
echo 虚拟环境激活命令（下次使用）：
echo   venv\Scripts\activate.bat
echo.

pause
