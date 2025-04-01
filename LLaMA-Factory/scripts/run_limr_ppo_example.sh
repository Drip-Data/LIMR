#!/bin/bash

# --- 配置区 ---
# 使用相对路径，基于脚本所在位置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"  # LLaMA-Factory 项目的根目录
# YAML 配置文件的路径 (确保指向您最终使用的 YAML 文件)
CONFIG_FILE="${PROJECT_ROOT}/examples/limr/limr_ppo_qwen1.5_example.yaml" # 使用优化后的 YAML 文件名
# 输出日志文件
LOG_FILE="${PROJECT_ROOT}/../limr_ppo_training.log"

# --- 创建必要的目录 ---
mkdir -p "${PROJECT_ROOT}/../saves/Qwen1.5-7B/limr_ppo_run1/samples"
echo "Created necessary directories for output"

# --- 执行命令 ---
cd "${PROJECT_ROOT}" || exit 1 # 切换目录，如果失败则退出

# --- 设置环境变量以增加日志详细程度 ---
export TRANSFORMERS_VERBOSITY=info
export ACCELERATE_LOG_LEVEL=info

# --- 分布式启动逻辑 ---
echo "Running in distributed mode using accelerate..."
echo "Training logs will be saved to: ${LOG_FILE}"

# 使用tee命令同时输出到终端和日志文件，移除了--verbose参数
accelerate launch src/train.py "${CONFIG_FILE}" 2>&1 | tee "${LOG_FILE}"

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
  echo "LIMR PPO training script finished successfully using config: ${CONFIG_FILE}"
else
  echo "LIMR PPO training script failed with exit code ${EXIT_CODE} using config: ${CONFIG_FILE}"
  echo "Please check the log file for details: ${LOG_FILE}"
fi

