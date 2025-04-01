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
from typing import Dict, List, Optional, Union

import torch
from ..extras import logging
from ..extras.packages import is_requests_available

if is_requests_available():
    import requests


logger = logging.get_logger(__name__)


class RemoteRewardClient:
    """
    A client for remote reward services.
    
    This client sends requests to a remote server to compute rewards for model-generated responses.
    """
    
    def __init__(self, reward_url: str):
        """
        Initialize the remote reward client.
        
        Args:
            reward_url: URL of the remote reward server.
        """
        if not is_requests_available():
            raise ImportError("The 'requests' package is required for RemoteRewardClient.")
        
        self.reward_url = reward_url
        logger.info_rank0(f"Initialized remote reward client with URL: {reward_url}")
    
    def get_rewards(
        self, 
        sample_ids: List[str], 
        queries: List[str], 
        responses: List[str]
    ) -> List[torch.Tensor]:
        """
        Get rewards from the remote server.
        
        Args:
            sample_ids: List of sample IDs.
            queries: List of queries/prompts.
            responses: List of model-generated responses.
            
        Returns:
            List of reward tensors.
        """
        try:
            # Prepare the payload
            payload = {
                "sample_ids": sample_ids,
                "queries": queries,
                "responses": responses
            }
            
            # Send the request
            headers = {"Content-Type": "application/json"}
            response = requests.post(self.reward_url, json=payload, headers=headers, timeout=30)
            
            # Check for errors
            response.raise_for_status()
            
            # Parse the response
            result = response.json()
            
            if "rewards" not in result:
                raise ValueError(f"Invalid response from reward server: {result}")
            
            # Convert rewards to tensors
            rewards = [torch.tensor([float(r)], dtype=torch.float) for r in result["rewards"]]
            
            return rewards
        
        except Exception as e:
            # Use warning_rank0 for consistency and to fix attribute error
            logger.warning_rank0(f"Error getting rewards from remote server: {e}") 
            # Return default negative rewards in case of error
            return [torch.tensor([-1.0], dtype=torch.float) for _ in range(len(sample_ids))]
