# LIMR 工作流

## 工作流程概述

1. **打开终端:** 导航到 `LLaMA-Factory-FOR-LIMR` 项目的根目录。

2. **运行脚本:**

   ```bash
   bash scripts/run_limr_ppo_example.sh
   ```

3. **脚本执行流程:**

   ### 阶段 1: 初始 PPO

   - 脚本会使用 `accelerate launch` (如果检测到多 GPU) 或 `python` 启动 `src/train.py`
   - 加载 YAML 配置
   - PPO 训练开始，模型生成响应
   - LIMR 奖励逻辑被调用计算奖励
   - `SampleWriter` 按照 `save_every_n_steps` 的频率将样本（包含奖励）保存到 `INITIAL_SAMPLES_DIR`

   ### 阶段 2: LIM 筛选

   - PPO 训练完成后，脚本调用 `LLaMA-Factory-FOR-LIMR\examples\limr\limr_analysis_llamafactory.py`
   - 该脚本读取 `INITIAL_SAMPLES_DIR` 中的所有 `.jsonl` 样本文件
   - 根据 `--threshold`进行筛选
   - 将筛选结果写入 `FILTERED_DATASET_INFO` 文件

   ### 阶段 3: SFT 微调

   - 再次运行 

     ```
     src/train.py
     ```

     ，通过命令行参数覆盖配置：

     - `--stage sft`: 设置为 SFT 阶段
     - `--dataset $FILTERED_DATASET_INFO`: 使用筛选后的数据集
     - `--output_dir $FINAL_OUTPUT_DIR`: 指定最终模型的保存位置
     - `--limr.enabled false`: 在 SFT 阶段通常禁用 LIMR 的奖励计算和样本保存
     - 可能调整学习率 (`--learning_rate`) 和训练周期 (`--num_train_epochs`)
     - 如果初始 PPO 训练保存了 LoRA 适配器，可能需要设置 `--adapter_name_or_path`

# 适配 LIMR 工作流所做修改的详细报告

### 1. 参数配置层 (`LIMRArguments` 和 `TrainingArguments`)

- **`LLaMA-Factory-FOR-LIMR/src/llamafactory/hparams/limr_args.py`**

  - **状态**: 新增文件

  - 内容: 定义了 

    ```
    LIMRArguments
    ```

     dataclass，包含 LIMR 特定参数

    - `enabled`: 启用/禁用 LIMR
    - `reward_type`: 奖励类型
    - `reward_url`: 远程奖励服务 URL
    - `ground_truth_dataset`: 基准数据集
    - `save_samples_path`: 样本保存路径
    - `save_every_n_steps`: 样本保存频率
    - `math_equal_normalize`: 数学表达式归一化
    - `reward_correct`: 正确答案奖励值
    - `reward_incorrect`: 错误答案奖励值

  - **目的**: 集中管理 LIMR 配置

- **`LLaMA-Factory-FOR-LIMR/src/llamafactory/hparams/training_args.py`**

  - **状态**: 修改文件
  - **修改**: 导入 `LIMRArguments` 并在 `TrainingArguments` 中添加 `limr: LIMRArguments` 字段
  - **目的**: 将 LIMR 参数集成到主训练参数体系

### 2. 数学工具函数 (`math_utils.py`)

- `LLaMA-Factory-FOR-LIMR/src/llamafactory/extras/math_utils.py`
  - **状态**: 新增/大幅修改文件
  - 内容: 实现多个数学处理函数
    - `extract_boxed_answer`: 提取盒装答案
    - `normalize_math_expression`: 数学表达式归一化
    - `math_equal`: 数学表达式等价性比较
  - **目的**: 提供 LIMR 规则奖励所需的数学处理能力

### 3. 数据处理层 (`data_utils.py`)

- `LLaMA-Factory-FOR-LIMR/src/llamafactory/data/data_utils.py`
  - **状态**: 修改文件
  - **修改**: 新增 `ensure_sample_ids` 函数，确保数据集中每个样本都有唯一的字符串 ID
  - **目的**: 为 LIMR 样本跟踪和关联提供基础

### 4. 奖励服务层 (`reward/` 目录)

- **`LLaMA-Factory-FOR-LIMR/src/llamafactory/reward/`**: 新增目录及 `__init__.py`

- **`LLaMA-Factory-FOR-LIMR/src/llamafactory/reward/rule_reward.py`**

  - **状态**: 新增文件

  - 内容: 定义 类

    ```
    RuleBasedRewardModel
    ```

    - 加载基准答案
    - 使用 `math_utils` 计算基于数学等价性的奖励

  - **目的**: 提供本地规则奖励计算

- **`LLaMA-Factory-FOR-LIMR/src/llamafactory/reward/remote_reward.py`**

  - **状态**: 新增文件
  - **内容**: 定义 `RemoteRewardClient` 类，通过 HTTP POST 与外部奖励服务通信
  - **目的**: 提供与远程奖励模型交互的能力

### 5. 样本保存机制 (`SampleWriter`)

- `LLaMA-Factory-FOR-LIMR/src/llamafactory/data/sample_writer.py`

  - **状态**: 新增文件

  - 内容: 定义 类

    ```
    SampleWriter
    ```

    - 在训练过程中将样本以 JSON Lines 格式写入文件 (`samples_step_*.jsonl`)
    - 样本包含: ID, prompt, response, reward, metadata
    - 支持分布式环境（仅 rank 0 写入）

  - **目的**: 持久化 PPO 阶段生成的样本供后续 LIM 分析

### 6. PPO 训练器集成 (`PPOTrainer`)

- `LLaMA-Factory-FOR-LIMR/src/llamafactory/train/ppo/trainer.py`
  - **状态**: 修改文件
  - 修改:
    - 修改 `get_rewards` 方法，在 `args.limr.enabled` 时调用新的 `_get_limr_rewards`
    - 新增 `_get_limr_rewards` 方法，根据 `args.limr.reward_type` 选择奖励计算方式
    - 新增 `_save_limr_samples` 方法，使用 `SampleWriter` 保存样本
    - 允许 `finetuning_args.reward_model_type` 设置为 `"limr"`
  - **目的**: 将 LIMR 奖励计算和样本保存逻辑集成到 PPO 训练循环

### 7. 数据筛选工具 (`limr_analysis_llamafactory.py`)

- `LLaMA-Factory-FOR-LIMR/examples/limr/limr_analysis_llamafactory.py`
  - **状态**: 新增文件
  - **内容**: 读取样本文件，根据奖励值筛选，并保存过滤后的样本信息
  - **目的**: 实现 LIMR 工作流的样本筛选步骤

### 8. 示例文件

- **`LLaMA-Factory-FOR-LIMR/examples/limr/limr_ppo_qwen1.5_example.yaml`**
  - **状态**: 新增文件
  - **内容**: 提供 LIMR PPO 训练的完整配置示例
  - **目的**: 展示 LIMR PPO 配置
- **`LLaMA-Factory-FOR-LIMR/scripts/run_limr_ppo_example.sh`**
  - **状态**: 新增文件
  - **内容**: 实现 LIMR 三步工作流脚本（初始 PPO -> LIM 筛选 -> SFT 微调）
  - **目的**: 提供端到端的 LIMR 工作流运行示例

# 目录结构对比分析

### 1. `src/llamafactory/` 核心代码目录

- **`hparams/`**:
  - `limr_args.py`: **新增** - 定义 LIMR 特定参数
  - `training_args.py`: **修改** - 导入并集成了 `limr_args.py`
- **`extras/`**:
  - `math_utils.py`: **新增/大幅修改** - 添加了数学答案提取和比较功能
- **`data/`**:
  - `data_utils.py`: **修改** - 添加了 `ensure_sample_ids` 函数
  - `sample_writer.py`: **新增** - 实现样本保存逻辑
- **`reward/`**:
  - **新增目录**
  - `__init__.py`: **新增**
  - `rule_reward.py`: **新增** - 定义 `RuleBasedRewardModel`
  - `remote_reward.py`: **新增** - 定义 `RemoteRewardClient`
- **`train/ppo/`**:
  - `trainer.py`: **修改** - 集成了 LIMR 奖励计算和样本保存逻辑

### 2. `examples/` 示例目录

- `limr/`:
  - **新增目录**
  - `limr_analysis_llamafactory.py`: **新增** - LIM 样本筛选脚本
  - `limr_ppo_qwen1.5_example.yaml`: **新增** - LIMR PPO 训练配置文件示例

### 3. `scripts/` 脚本目录

- `run_limr_ppo_example.sh`: **新增** - 端到端的 LIMR 工作流运行脚本示例

### 4. `tests/` 测试目录

- `__init__.py`: **新增/确认存在** - 确保 `tests` 是一个包
- `test_math_utils.py`: **新增** - 针对 `math_utils.py` 的单元测试
- `test_rule_reward.py`: **新增** - 针对 `RuleBasedRewardModel` 的单元测试
- `test_sample_writer.py`: **新增** - 针对 `SampleWriter` 的单元测试
- `test_remote_reward.py`: **新增** - 针对 `RemoteRewardClient` 的单元测试

# 核心文件：PPO/trainer.py 的修改说明

以下是 `LLaMA-Factory-FOR-LIMR/src/llamafactory/train/ppo/trainer.py` 中为适配 LIMR 工作流所做的核心代码修改情况说明：

## 1. `get_rewards` 方法修改

- 增加 LIMR 条件分支
  - 在该方法的核心逻辑之前，添加了 `if self.args.limr.enabled:` 的判断
  - 如果 `args.limr.enabled` 为 `True`，则直接调用 `return self._get_limr_rewards(queries, responses)`
  - 如果 `args.limr.enabled` 为 `False`，则执行原有的基于 `self.finetuning_args.reward_model_type` 的奖励计算流程
- 目的
  - 实现奖励计算流程的条件切换
  - 根据配置决定是使用标准 PPO 奖励还是 LIMR 特定奖励

## 2. 新增 `_get_limr_rewards` 方法

- 功能

  - 封装 LIMR 的奖励计算和样本保存触发逻辑

- 核心步骤

  - 从批次数据中提取 `sample_ids`

  - 使用 `tokenizer.batch_decode` 解码 `queries` 和 `responses` 为文本

  - 根据 

    ```
    self.args.limr.reward_type
    ```

     的值 ("rule" 或 "remote")：

    - **动态实例化奖励模型**：如果对应的奖励模型实例尚未创建，则根据配置进行实例化
    - **调用奖励计算**：调用实例化后的奖励模型的 `get_rewards` 方法

  - **触发样本保存**：检查是否满足样本保存条件，如果满足，则调用 `self._save_limr_samples`

  - 返回计算得到的奖励 Tensor 列表

- 目的

  - 将 LIMR 的特定处理流程（奖励选择、计算、样本保存触发）集中管理

## 3. 新增 `_save_limr_samples` 方法

- 功能
  - 封装使用 `SampleWriter` 保存样本的逻辑
- 核心步骤
  - **动态实例化 `SampleWriter`**：如果尚未创建，则使用配置参数进行实例化
  - **准备数据**：整理 `sample_ids`, `queries`, `responses`, `rewards` 以及可选的元数据
  - **调用写入**：调用 `self._limr_sample_writer.write_batch` 将数据写入 JSON Lines 文件
- 目的
  - 实现将 PPO 过程中产生的关键信息保存，供后续分析

## 4. 处理 `reward_model_type: limr`

- 整个类的逻辑现在能识别 `finetuning_args.reward_model_type == "limr"`
- 当此设置生效时，`get_rewards` 方法会通过 LIMR 分支处理奖励
- 这种情况下通常不会尝试加载或使用 `self.reward_model` 属性
- 目的
  - 使框架能够通过配置识别并启用 LIMR 的奖励流程
  - 避免强制加载一个标准的外部奖励模型

### 总结

对 `ppo/trainer.py` 的修改核心在于通过条件判断和新增辅助方法，将 LIMR 的奖励计算（规则或远程）和样本保存功能"注入"到 PPO 的训练循环中，同时保持了与原有 PPO 逻辑的兼容性。

