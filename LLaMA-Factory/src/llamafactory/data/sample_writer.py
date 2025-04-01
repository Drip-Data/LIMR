# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import threading
from typing import Dict, List, Any, Optional

from ..extras import logging

logger = logging.get_logger(__name__)

class SampleWriter:
    """训练样本保存器，用于在训练过程中保存生成的样本"""
    
    def __init__(self, save_path: str, distributed: bool = False, rank: int = 0):
        """
        初始化样本保存器
        
        Args:
            save_path: 保存路径
            distributed: 是否在分布式环境中运行
            rank: 当前进程的rank (分布式环境中使用)
        """
        self.save_path = save_path
        self.distributed = distributed
        self.rank = rank
        self.lock = threading.Lock()
        
        # 创建保存目录
        os.makedirs(save_path, exist_ok=True)
        logger.info_rank0(f"Initialized sample writer at: {save_path}")
    
    def write_batch(
        self,
        step: int,
        sample_ids: List[str],
        prompts: List[str],
        responses: List[str],
        rewards: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        保存一批样本
        
        Args:
            step: 当前训练步数
            sample_ids: 样本ID列表
            prompts: 提示/查询列表
            responses: 模型生成的回复列表
            rewards: 奖励值列表
            metadata: 额外的元数据信息
        """
        if self.distributed and self.rank != 0:
            # 在分布式环境中，只有rank 0进程保存样本
            return
        
        with self.lock:
            # 构建文件名
            filename = f"samples_step_{step}.jsonl"
            filepath = os.path.join(self.save_path, filename)
            
            # 构建样本数据
            samples = []
            for i in range(len(sample_ids)):
                sample = {
                    "id": sample_ids[i],
                    "prompt": prompts[i],
                    "response": responses[i],
                    # Convert tensor to float for JSON serialization
                    "reward": rewards[i].item() if hasattr(rewards[i], 'item') else rewards[i] 
                }
                
                # 添加元数据（如果有）
                if metadata is not None:
                    for key, value in metadata.items():
                        # Check if value is a sequence and index is valid
                        if isinstance(value, (list, tuple)) and i < len(value):
                            sample[key] = value[i]
                        # If it's not a sequence, assume it applies to all samples in the batch
                        elif not isinstance(value, (list, tuple)):
                             sample[key] = value
                        # Handle cases where sequence is too short (optional, maybe log a warning)
                        # else:
                        #     logger.warning_rank0(f"Metadata '{key}' sequence shorter than batch size.")
                
                samples.append(sample)
            
            # 写入文件
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    for sample in samples:
                        f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                logger.info_rank0(f"Saved {len(samples)} samples to {filepath}")
            except Exception as e:
                # Use warning_rank0 for consistency and to fix attribute error
                logger.warning_rank0(f"Error saving samples to {filepath}: {e}") 
    
    def write_sample(
        self,
        step: int,
        sample_id: str,
        prompt: str,
        response: str,
        reward: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        保存单个样本
        
        Args:
            step: 当前训练步数
            sample_id: 样本ID
            prompt: 提示/查询
            response: 模型生成的回复
            reward: 奖励值
            metadata: 额外的元数据信息
        """
        self.write_batch(
            step=step,
            sample_ids=[sample_id],
            prompts=[prompt],
            responses=[response],
            rewards=[reward],
            metadata={k: [v] for k, v in metadata.items()} if metadata else None
        )
