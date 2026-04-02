#!/bin/bash

# ============================================================================
# 人脸检测与去重流水线 - 智能安装脚本
# ============================================================================

set -e  # 遇到错误立即退出

echo "========================================================================"
echo "人脸检测与去重流水线 - 智能安装向导"
echo "========================================================================"
echo ""

# ============================================================================
# 第1步：检测已有环境
# ============================================================================

echo "[步骤1] 环境检测..."
echo "  检查虚拟环境..."

USE_CONDA=false
USE_VENV=false
ENV_EXISTS=false

# 检查是否已有激活的conda环境
if command -v conda &> /dev/null; then
    CONDA_DEFAULT_ENV=$(conda info --json | grep -o '"default_prefix": "[^"]*' | cut -d'"' -f4)
    if [ ! -z "$CONDA_DEFAULT_ENV" ] && [ ! "$CONDA_DEFAULT_ENV" = "None" ]; then
        echo "  ✅ 检测到conda已安装"
        
        # 检查是否已有face_detect环境
        if conda env list | grep -q "face_detect"; then
            echo "  ✅ 检测到已有'face_detect' conda环境"
            ENV_EXISTS=true
            USE_CONDA=true
        else
            echo "  ℹ️  未找到'face_detect' conda环境"
        fi
    fi
fi

# 检查虚拟环境
if [ -d "venv" ]; then
    echo "  ✅ 检测到本地虚拟环境 (./venv)"
    ENV_EXISTS=true
    USE_VENV=true
fi

echo ""
if [ "$ENV_EXISTS" = true ]; then
    echo "✅ 已有环境存在"
    read -p "是否使用现有环境？(y/n) [默认: y]: " use_existing
    use_existing=${use_existing:-y}
    
    if [ "$use_existing" != "y" ]; then
        ENV_EXISTS=false
    fi
fi

echo ""

# ============================================================================
# 第2步：环境创建（如果不存在）
# ============================================================================

if [ "$ENV_EXISTS" = false ]; then
    echo "[步骤2] 创建环境..."
    echo ""
    echo "选择环境管理工具："
    echo "  1) conda (推荐，使用environment.yml)"
    echo "  2) venv (默认，使用requirements.txt)"
    echo ""
    read -p "请选择 [1-2，默认2]: " env_choice
    env_choice=${env_choice:-2}
    
    if [ "$env_choice" = "1" ]; then
        echo ""
        echo "使用conda创建环境..."
        
        # 检查conda是否可用
        if ! command -v conda &> /dev/null; then
            echo "❌ 错误：未找到conda"
            echo "请先安装Anaconda或Miniconda: https://www.anaconda.com/download"
            exit 1
        fi
        
        # 创建conda环境
        if [ -f "environment.yml" ]; then
            echo "从environment.yml创建环境..."
            conda env create -f environment.yml -y
            ENV_NAME=$(grep '^name:' environment.yml | sed 's/name: //')
            conda activate $ENV_NAME
            USE_CONDA=true
        else
            echo "❌ 错误：未找到environment.yml文件"
            exit 1
        fi
    else
        echo ""
        echo "使用venv创建虚拟环境..."
        python3 -m venv venv
        
        # 激活虚拟环境
        if [ -f "venv/bin/activate" ]; then
            source venv/bin/activate
        elif [ -f "venv/Scripts/activate" ]; then
            source venv/Scripts/activate
        else
            echo "❌ 错误：无法激活虚拟环境"
            exit 1
        fi
        USE_VENV=true
    fi
    
    echo "✅ 环境创建成功"
else
    echo "[步骤2] 激活现有环境..."
    
    if [ "$USE_CONDA" = true ]; then
        ENV_NAME=$(conda env list | grep "face_detect" | awk '{print $1}')
        if [ ! -z "$ENV_NAME" ]; then
            conda activate $ENV_NAME
            echo "✅ 已激活conda环境: $ENV_NAME"
        fi
    elif [ "$USE_VENV" = true ]; then
        if [ -f "venv/bin/activate" ]; then
            source venv/bin/activate
        elif [ -f "venv/Scripts/activate" ]; then
            source venv/Scripts/activate
        fi
        echo "✅ 已激活虚拟环境"
    fi
fi

echo ""

# ============================================================================
# 第3步：升级pip（仅限venv）
# ============================================================================

if [ "$USE_VENV" = true ]; then
    echo "[步骤3] 升级pip和setuptools..."
    pip install --upgrade pip setuptools wheel --quiet
    echo "✅ pip已升级"
else
    echo "[步骤3] 跳过pip升级（使用conda管理）"
fi

echo ""

# ============================================================================
# 第4步：安装PyTorch（仅限venv）
# ============================================================================

if [ "$USE_VENV" = true ]; then
    echo "[步骤4] 安装PyTorch..."
    echo ""
    echo "请根据你的系统选择CUDA版本："
    echo ""
    echo "  1) CUDA 12.x (推荐)"
    echo "  2) CUDA 11.8"
    echo "  3) CPU only (无CUDA)"
    echo "  4) Apple Silicon (M1/M2/M3)"
    echo ""
    read -p "请选择 [1-4，默认1]: " cuda_choice
    cuda_choice=${cuda_choice:-1}
    
    case $cuda_choice in
        1)
            echo "安装 PyTorch CUDA 12.x ..."
            pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet
            ;;
        2)
            echo "安装 PyTorch CUDA 11.8 ..."
            pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --quiet
            ;;
        3)
            echo "安装 PyTorch CPU ..."
            pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet
            ;;
        4)
            echo "安装 PyTorch Apple Silicon ..."
            pip install torch torchvision --quiet
            ;;
        *)
            echo "❌ 无效的选择，使用默认CUDA 12.x"
            pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet
            ;;
    esac
    
    echo "✅ PyTorch安装完成"
else
    echo "[步骤4] 跳过PyTorch安装（已包含在environment.yml中）"
fi

echo ""

# ============================================================================
# 第5步：安装其他依赖
# ============================================================================

echo "[步骤5] 安装项目依赖..."
echo ""

if [ "$USE_CONDA" = true ]; then
    echo "使用environment.yml安装依赖（已完成）"
    echo "如需更新依赖，运行："
    echo "  conda env update -f environment.yml -y"
else
    echo "从requirements.txt安装依赖..."
    
    # 检查是否使用国内镜像
    read -p "是否使用国内镜像源加速？(y/n) [默认n]: " use_mirror
    use_mirror=${use_mirror:-n}
    
    if [ "$use_mirror" = "y" ]; then
        echo "使用阿里云镜像..."
        pip install -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt
    else
        echo "使用官方源..."
        pip install -r requirements.txt
    fi
fi

echo "✅ 依赖安装完成"

echo ""

# ============================================================================
# 第6步：下载预训练模型
# ============================================================================

echo "[步骤6] 下载预训练模型..."
echo ""
echo "这将下载 InsightFace 和 YOLOv11-face 模型（约375MB）"
echo "首次下载可能需要几分钟，请耐心等待..."
echo ""

read -p "现在下载模型吗？(y/n) [默认y]: " download_models
download_models=${download_models:-y}

if [ "$download_models" = "y" ]; then
    python download_models.py
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ 模型下载完成"
    else
        echo ""
        echo "⚠️  模型下载出现问题，但不影响使用"
        echo "系统会在首次运行时尝试自动下载"
    fi
else
    echo "⏭️  跳过模型下载"
    echo "注意：首次运行时可能需要自动下载模型"
fi

echo ""

# ============================================================================
# 第7步：完成
# ============================================================================

echo "========================================================================">
echo "✅ 安装完成！"
echo "========================================================================"
echo ""
echo "现在你可以运行以下命令之一来开始处理视频："
echo ""
echo "基本使用（处理单个视频）："
echo "  python face_dedup_pipeline.py input_video.mp4"
echo ""
echo "指定输出目录："
echo "  python face_dedup_pipeline.py input_video.mp4 -o ./output"
echo ""
echo "使用GPU加速："
echo "  python face_dedup_pipeline.py input_video.mp4 --cuda"
echo ""
echo "处理视频目录："
echo "  python face_dedup_pipeline.py ./videos/ -o ./output"
echo ""
echo "查看所有可用参数："
echo "  python face_dedup_pipeline.py --help"
echo ""

if [ "$USE_CONDA" = true ]; then
    echo "下次使用时激活环境："
    echo "  conda activate $ENV_NAME"
elif [ "$USE_VENV" = true ]; then
    echo "下次使用时激活环境："
    echo "  source venv/bin/activate"
fi

echo ""
echo "========================================================================"
