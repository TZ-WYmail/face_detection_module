#!/usr/bin/env python3
"""
项目初始化脚本

功能：
1. 创建完整的项目目录结构
2. 下载模型到项目本地
3. 生成配置文件和快速开始指南
4. 验证环境和依赖

使用方法：
    python setup_project.py
    python setup_project.py --skip-models
"""

from __future__ import annotations
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, List
import subprocess

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProjectSetup:
    """项目初始化工具"""

    # 项目结构定义
    PROJECT_DIRS = {
        'input': {
            'name': 'videos',
            'desc': '输入视频文件',
            'subdirs': []
        },
        'output': {
            'name': 'detected_faces_frontal',
            'desc': '检测出的正脸图像',
            'subdirs': []
        },
        'data': {
            'name': 'data',
            'desc': '数据文件（临时缓存、中间结果）',
            'subdirs': ['temp', 'cache']
        },
        'models': {
            'name': 'models',
            'desc': '本地模型文件',
            'subdirs': ['insightface', 'yolo', 'reid']
        },
        'logs': {
            'name': 'logs',
            'desc': '运行日志',
            'subdirs': []
        },
        'config': {
            'name': 'config',
            'desc': '配置文件',
            'subdirs': []
        }
    }

    def __init__(self, project_root: Optional[str] = None):
        """
        初始化项目设置

        Args:
            project_root: 项目根目录（默认：当前目录）
        """
        self.project_root = Path(project_root or '.')
        self.project_root = self.project_root.resolve()
        logger.info(f"项目根目录: {self.project_root}")

    def create_directory_structure(self) -> bool:
        """创建完整的项目目录结构"""
        logger.info("\n" + "="*60)
        logger.info("创建项目目录结构")
        logger.info("="*60)

        try:
            for key, info in self.PROJECT_DIRS.items():
                dir_path = self.project_root / info['name']
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"✅ {info['name']:20s} - {info['desc']}")

                # 创建子目录
                for subdir in info['subdirs']:
                    subdir_path = dir_path / subdir
                    subdir_path.mkdir(parents=True, exist_ok=True)
                    logger.info(f"   ├── {subdir}")

            logger.info("\n✅ 目录结构创建完成")
            return True
        except Exception as e:
            logger.error(f"创建目录失败: {e}")
            return False

    def download_reid_model(self) -> bool:
        """下载 OSNet ReID 模型到 models/reid/ 目录"""
        logger.info("\n" + "="*60)
        logger.info("下载 ReID (OSNet) 模型")
        logger.info("="*60)

        reid_dir = self.project_root / 'models' / 'reid'
        reid_dir.mkdir(parents=True, exist_ok=True)

        model_path = reid_dir / 'osnet_x0_25_msmt17.pt'
        if model_path.exists():
            size_mb = model_path.stat().st_size / 1024 / 1024
            logger.info(f"✅ ReID 模型已存在: {model_path} ({size_mb:.2f} MB)")
            return True

        logger.info("模型不存在，开始下载 OSNet-x0.25 (MSMT17 预训练) ...")
        logger.info("模型大小: ~3 MB，首次运行会自动从 torchreid 下载预训练权重")

        try:
            script = '''
import sys, os
try:
    from torchreid import models
    import torch

    # 构建 OSNet 模型并加载 MSMT17 预训练权重
    model = models.build_model(
        name="osnet_x0_25",
        num_classes=10429,
        pretrained="msmt17"
    )
    model.eval()

    # 测试推理
    x = torch.randn(1, 3, 256, 128)
    with torch.no_grad():
        feat = model(x)
    feat = feat / (torch.norm(feat, p=2) + 1e-12)
    print(f"ReID embedding shape: {feat.shape}")

    # 保存为 TorchScript
    if torch.cuda.is_available():
        model = model.cuda()
        x = x.cuda()
        scripted = torch.jit.trace(model, x)
    else:
        scripted = torch.jit.trace(model, x)

    save_path = "''' + str(model_path) + '''"
    scripted.save(save_path)
    size_mb = os.path.getsize(save_path) / 1024 / 1024
    print(f"Saved: {save_path} ({size_mb:.2f} MB)")
    print("SUCCESS")
except ImportError as e:
    print(f"MISSING_DEP: {e}")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
'''
            result = subprocess.run(
                [sys.executable, '-c', script],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode != 0:
                output = result.stdout + result.stderr
                if 'MISSING_DEP' in output:
                    logger.error(
                        "❌ 缺少依赖: torchreid。请先安装:\n"
                        "   pip install torchreid"
                    )
                else:
                    logger.error(f"❌ ReID 模型下载失败:\n{output[:2000]}")
                return False

            if 'SUCCESS' in result.stdout:
                size_mb = model_path.stat().st_size / 1024 / 1024
                logger.info(f"✅ ReID 模型下载成功: {model_path} ({size_mb:.2f} MB)")
                return True
            else:
                logger.warning(f"⚠️ 模型下载结果不确定:\n{result.stdout[:1000]}")
                return model_path.exists()

        except subprocess.TimeoutExpired:
            logger.error("❌ ReID 模型下载超时（5分钟）")
            return False
        except Exception as e:
            logger.error(f"❌ ReID 模型下载失败: {e}")
            return False

    def download_models(self, hf_token: Optional[str] = None) -> bool:
        """下载模型到项目的 models 文件夹"""
        logger.info("\n" + "="*60)
        logger.info("下载模型文件")
        logger.info("="*60)

        try:
            # 使用 download_models.py 脚本下载到项目的 models 目录
            models_dir = self.project_root / 'models'
            insightface_dir = models_dir / 'insightface' / 'models'
            yolo_dir = models_dir / 'yolo'

            cmd = [
                sys.executable,
                'download_models.py',
                '--insightface-dir', str(insightface_dir),
                '--yolo-dir', str(yolo_dir)
            ]

            if hf_token:
                cmd.extend(['--hf-token', hf_token])

            logger.info(f"执行命令: {' '.join(cmd)}")
            result = subprocess.run(cmd, cwd=self.project_root)

            if result.returncode == 0:
                logger.info("✅ 模型下载成功")
                return True
            else:
                logger.warning("⚠️ 模型下载有问题，请查看上面的输出")
                return False

        except Exception as e:
            logger.error(f"模型下载失败: {e}")
            return False

    def create_env_file(self) -> bool:
        """创建项目配置文件"""
        logger.info("\n" + "="*60)
        logger.info("创建配置文件")
        logger.info("="*60)

        try:
            env_path = self.project_root / '.env'
            env_content = f"""# 项目环境配置文件

# 模型路径（相对于项目根目录）
INSIGHTFACE_HOME={(self.project_root / 'models' / 'insightface' / 'models').relative_to(self.project_root)}
YOLO_HOME={(self.project_root / 'models' / 'yolo').relative_to(self.project_root)}

# 输入输出路径
VIDEO_INPUT_DIR=videos
FACES_OUTPUT_DIR=detected_faces_frontal
DATA_TEMP_DIR=data/temp

# HuggingFace Token (可选，用于加速模型下载)
# HF_TOKEN=hf_xxxxxxxxxxxx

# 日志设置
LOG_DIR=logs
LOG_LEVEL=INFO

# 处理参数
DEFAULT_CUDA=true
SAMPLE_INTERVAL=1
CONFIDENCE=0.5
"""
            with open(env_path, 'w', encoding='utf-8') as f:
                f.write(env_content)
            logger.info(f"✅ 配置文件创建: {env_path}")
            return True
        except Exception as e:
            logger.error(f"创建配置文件失败: {e}")
            return False

    def create_quick_start_guide(self) -> bool:
        """创建快速开始指南"""
        logger.info("\n" + "="*60)
        logger.info("创建快速开始指南")
        logger.info("="*60)

        try:
            guide_path = self.project_root / 'QUICKSTART.md'
            guide_content = """# 项目快速开始指南

## 📁 项目结构

```
.
├── videos/                      # 📹 输入视频文件（放你的视频在这里）
├── detected_faces_frontal/      # 📸 输出：检测的正脸图像
├── data/
│   ├── temp/                   # 临时文件
│   └── cache/                  # 缓存文件
├── models/                      # 🤖 模型文件
│   ├── insightface/
│   └── yolo/
├── logs/                        # 📝 运行日志
├── config/                      # ⚙️ 配置文件
├── face_dedup_pipeline.py       # 主程序
├── face_dedup_utils.py          # 工具库
├── config.py                    # 参数配置
└── .env                         # 环境变量
```

## 🚀 快速开始

### 1️⃣ 准备工作

放置你的视频文件到 `videos/` 文件夹：

```bash
cp your_video.mp4 videos/
```

### 2️⃣ 基本用法

**处理单个视频（推荐使用 GPU）：**

```bash
python face_dedup_pipeline.py videos/your_video.mp4 --cuda
```

**处理整个 videos 文件夹：**

```bash
python face_dedup_pipeline.py videos --cuda
```

**仅使用 CPU（不依赖 GPU）：**

```bash
python face_dedup_pipeline.py videos/your_video.mp4
```

### 3️⃣ 查看结果

处理完后，检测的正脸图像会保存到：

```bash
detected_faces_frontal/
├── video_name_001.jpg
├── video_name_002.jpg
└── ...
```

## 📊 常用参数

### 高质量模式（身份证照）

```bash
python face_dedup_pipeline.py videos/your_video.mp4 --cuda \\
    --yaw-threshold 10 \\
    --pitch-threshold 10 \\
    --roll-threshold 5 \\
    --confidence 0.7 \\
    --threshold 0.6
```

### 快速处理模式（视频分析）

```bash
python face_dedup_pipeline.py videos/your_video.mp4 --cuda \\
    --sample-interval 5 \\
    --detector yolo \\
    --confidence 0.3
```

### 查看所有参数

```bash
python face_dedup_pipeline.py --help
```

## ⚙️ 配置文件

编辑 `.env` 文件以修改默认配置：

```ini
# 设置是否默认使用 GPU
DEFAULT_CUDA=true

# 采样间隔（每隔 N 帧处理一次，越大处理越快）
SAMPLE_INTERVAL=1

# 检测置信度（0-1，越高越严格）
CONFIDENCE=0.5
```

## 🔧 高级配置

参考 `config.py` 文件了解所有可配置参数：

```bash
cat config.py
```

## 📚 更多帮助

- 完整文档：见 [README.md](README.md)
- 安装指南：见 [SETUP.md](SETUP.md)
- 架构说明：见 [ARCHITECTURE.md](ARCHITECTURE.md)
- 参数详解：见 [config.py](config.py)

## ⚡ 常见问题

### Q: 处理速度太慢？

**Ans:**
1. 启用 GPU：`--cuda`
2. 增加采样间隔：`--sample-interval 5`
3. 切换到轻量检测器：`--detector yolo`

### Q: 显存不足？

**Ans:**
1. 使用 CPU：移除 `--cuda`
2. 增加采样间隔减少内存消耗
3. 处理分辨率更低的视频

### Q: 如何修改输出目录？

**Ans:**
```bash
python face_dedup_pipeline.py videos/your_video.mp4 --output custom_output_dir
```

### Q: 找不到模型？

**Ans:**
检查 `models/` 目录是否包含：
```bash
ls models/insightface/models/buffalo_l/
ls models/yolo/
```

如果缺失，重新运行：
```bash
python download_models.py --all-dirs ./models
```

## 💡 工作流示例

```bash
# 1. 放置视频
cp /path/to/video.mp4 videos/

# 2. 处理视频
python face_dedup_pipeline.py videos/video.mp4 --cuda

# 3. 查看结果
ls -lh detected_faces_frontal/

# 4. 查看日志
tail -f logs/*.log  # 如果有日志输出
```

## 📞 获取帮助

- 查看 `--help`：`python face_dedup_pipeline.py --help`
- 查看日志：`logs/` 目录
- 查看完整文档：[README.md](README.md)

---

**现在就开始：**
```bash
python face_dedup_pipeline.py videos/ --cuda
```

祝你使用愉快！ 🎉
"""
            with open(guide_path, 'w', encoding='utf-8') as f:
                f.write(guide_content)
            logger.info(f"✅ 快速开始指南: {guide_path}")
            return True
        except Exception as e:
            logger.error(f"创建快速开始指南失败: {e}")
            return False

    def create_example_script(self) -> bool:
        """创建示例脚本"""
        logger.info("\n" + "="*60)
        logger.info("创建示例脚本")
        logger.info("="*60)

        try:
            script_path = self.project_root / 'example_run.sh'
            script_content = """#!/bin/bash

# 项目使用示例脚本

echo "=========================================="
echo "人脸检测与去重流水线 - 示例运行"
echo "=========================================="
echo ""

# 设置项目根目录
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR" || exit 1

# 示例 1: 处理单个视频（GPU）
echo "示例 1: 处理单个视频（GPU 加速）"
echo "命令: python face_dedup_pipeline.py videos/your_video.mp4 --cuda"
echo ""
# python face_dedup_pipeline.py videos/your_video.mp4 --cuda

# 示例 2: 处理整个文件夹（CPU）
echo "示例 2: 处理整个文件夹（CPU）"
echo "命令: python face_dedup_pipeline.py videos/ --output custom_output"
echo ""
# python face_dedup_pipeline.py videos/ --output custom_output

# 示例 3: 高质量模式（身份证照采集）
echo "示例 3: 高质量模式（身份证照采集）"
echo "命令: python face_dedup_pipeline.py videos/id_photo.mp4 --cuda \\\\"
echo "        --yaw-threshold 10 --pitch-threshold 10 --roll-threshold 5"
echo ""
# python face_dedup_pipeline.py videos/id_photo.mp4 --cuda \\
#     --yaw-threshold 10 --pitch-threshold 10 --roll-threshold 5

# 示例 4: 快速处理模式
echo "示例 4: 快速处理模式（速度优先）"
echo "命令: python face_dedup_pipeline.py videos/ --cuda \\\\"
echo "        --detector yolo --sample-interval 5 --confidence 0.3"
echo ""
# python face_dedup_pipeline.py videos/ --cuda \\
#     --detector yolo --sample-interval 5 --confidence 0.3

echo ""
echo "=========================================="
echo "提示："
echo "  • 将你的视频放到 videos/ 文件夹"
echo "  • 取消注释（删除 #）以运行相应示例"
echo "  • 结果保存到 detected_faces_frontal/ 文件夹"
echo "  • 查看所有参数：python face_dedup_pipeline.py --help"
echo "=========================================="
"""
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(script_content)
            # 添加执行权限
            os.chmod(script_path, 0o755)
            logger.info(f"✅ 示例脚本: {script_path}")
            return True
        except Exception as e:
            logger.error(f"创建示例脚本失败: {e}")
            return False

    def print_project_summary(self) -> None:
        """打印项目初始化总结"""
        logger.info("\n" + "="*60)
        logger.info("项目初始化完成！")
        logger.info("="*60)
        logger.info(f"\n📁 项目位置: {self.project_root}")
        logger.info("\n📂 项目结构:")
        for key, info in self.PROJECT_DIRS.items():
            dir_path = self.project_root / info['name']
            logger.info(f"  ✅ {info['name']:20s} - {info['desc']}")

        logger.info("\n📚 生成的文件:")
        logger.info(f"  ✅ .env                - 环境配置文件")
        logger.info(f"  ✅ QUICKSTART.md       - 快速开始指南")
        logger.info(f"  ✅ example_run.sh      - 示例脚本")

        logger.info("\n🚀 下一步:")
        logger.info(f"  1. 放置视频: cp your_video.mp4 {self.project_root}/videos/")
        logger.info(f"  2. 运行程序: python face_dedup_pipeline.py videos/your_video.mp4 --cuda")
        logger.info(f"  3. 查看结果: ls {self.project_root}/detected_faces_frontal/")

        logger.info("\n📖 查看文档:")
        logger.info(f"  • 快速开始: cat QUICKSTART.md")
        logger.info(f"  • 完整文档: cat README.md")
        logger.info(f"  • 架构说明: cat ARCHITECTURE.md")

        logger.info("\n💡 帮助:")
        logger.info(f"  python face_dedup_pipeline.py --help")

        logger.info("\n✨ 祝你使用愉快！\n")

    def run(self, skip_models: bool = False, hf_token: Optional[str] = None) -> bool:
        """运行完整的项目初始化流程"""
        logger.info("\n" + "="*60)
        logger.info("开始项目初始化")
        logger.info("="*60)

        # 1. 创建目录结构
        if not self.create_directory_structure():
            return False

        # 2. 下载模型
        if not skip_models:
            if not self.download_models(hf_token):
                logger.warning("⚠️ 模型下载失败，但项目结构已创建")            # 下载 ReID 模型
            self.download_reid_model()        else:
            logger.info("⏭️  跳过模型下载（--skip-models）")

        # 3. 创建配置文件
        if not self.create_env_file():
            return False

        # 4. 创建快速开始指南
        if not self.create_quick_start_guide():
            return False

        # 5. 创建示例脚本
        if not self.create_example_script():
            return False

        # 6. 打印总结
        self.print_project_summary()

        return True


def main():
    parser = argparse.ArgumentParser(
        description='人脸检测项目初始化工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 完整初始化（包括下载模型）
  python setup_project.py

  # 仅创建目录结构，跳过模型下载
  python setup_project.py --skip-models

  # 指定项目目录
  python setup_project.py --project-dir /path/to/project

  # 使用 HF Token 加速下载
  export HF_TOKEN=hf_xxxxx
  python setup_project.py
"""
    )

    parser.add_argument('--project-dir', type=str, default='.',
                        help='项目根目录（默认：当前目录）')
    parser.add_argument('--skip-models', action='store_true',
                        help='跳过模型下载，仅创建目录结构')
    parser.add_argument('--hf-token', type=str, default=None,
                        help='HuggingFace Token（可选，用于加速下载）')

    args = parser.parse_args()

    # 如果没有提供 Token，尝试从环境变量读取
    hf_token = args.hf_token or os.environ.get('HF_TOKEN')

    setup = ProjectSetup(project_root=args.project_dir)
    success = setup.run(skip_models=args.skip_models, hf_token=hf_token)

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
