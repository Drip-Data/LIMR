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

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class LIMRArguments:
    """
    Arguments pertaining to the LIMR (Learning Impact Measurement for Reinforcement) workflow.
    """
    
    enabled: bool = field(
        default=False,
        metadata={"help": "Whether to enable LIMR workflow enhancements."}
    )
    
    reward_type: Literal["rule", "remote"] = field(
        default="rule",
        metadata={"help": "Type of reward mechanism ('rule' or 'remote')."}
    )
    
    reward_url: Optional[str] = field(
        default=None,
        metadata={"help": "URL of the remote reward server if reward_type is 'remote'."}
    )
    
    ground_truth_dataset: Optional[str] = field(
        default=None,
        metadata={"help": "Path or name of the dataset containing ground truth answers for rule-based reward."}
    )
    
    save_samples_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save generated samples per step."}
    )
    
    save_every_n_steps: int = field(
        default=1,
        metadata={"help": "Frequency (in steps) to save generated samples."}
    )
    
    math_equal_normalize: bool = field(
        default=True,
        metadata={"help": "Whether to normalize mathematical expressions before comparison."}
    )
    
    reward_correct: float = field(
        default=1.0,
        metadata={"help": "Reward value for correct answers."}
    )
    
    reward_incorrect: float = field(
        default=-1.0,
        metadata={"help": "Reward value for incorrect answers."}
    )