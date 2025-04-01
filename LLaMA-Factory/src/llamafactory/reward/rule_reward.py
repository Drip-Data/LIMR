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
import re # Import re for answer extraction fallback
from typing import Dict, List, Optional, Any, Tuple # Adjusted imports

import torch
from datasets import load_dataset, Dataset, VerificationMode # Ensure Dataset and VerificationMode are imported
from transformers.utils import is_datasets_available # Import is_datasets_available

from ..extras import logging # Keep this one for the logger definition below
from ..extras.math_utils import extract_boxed_answer, math_equal


# Remove the incorrect relative import
# from . import logging
logger = logging.get_logger(__name__)

# --- Add UUID Normalization Function ---
def normalize_uuid(id_str):
    """规范化UUID格式，确保一致性"""
    if not id_str or not isinstance(id_str, str):
        return str(id_str) if id_str is not None else "" # Return empty string for None
    # 移除可能的空格和转为小写
    return id_str.strip().lower()
# --- End UUID Normalization Function ---


class RuleBasedRewardModel:
    """
    A rule-based reward model that computes rewards based on mathematical equality.
    Loads ground truth answers and uses sample IDs for matching.
    Handles potential mismatch between sequential IDs and original UUIDs via mapping.
    """

    def __init__(
        self,
        ground_truth_dataset: str,
        correct_reward: float = 1.0,
        incorrect_reward: float = -1.0,
        normalize: bool = True,
    ):
        """
        Initialize the rule-based reward model.
        """
        self.uuid_to_data, self.sequential_to_uuid = self._load_ground_truth_and_create_mapping(ground_truth_dataset)
        self.correct_reward = correct_reward
        self.incorrect_reward = incorrect_reward
        self.normalize = normalize
        self._debug_logged = False # Flag for one-time debug log

        logger.info_rank0(f"Loaded {len(self.uuid_to_data)} ground truth answers from {ground_truth_dataset}.")
        if len(self.uuid_to_data) > 0:
             keys_sample = list(self.uuid_to_data.keys())[:5]
             logger.info_rank0(f"Ground truth dataset normalized UUID samples: {keys_sample}")
        if len(self.sequential_to_uuid) > 0:
             seq_keys_sample = list(self.sequential_to_uuid.items())[:5]
             logger.info_rank0(f"Sequential ID to UUID mapping samples: {seq_keys_sample}")


    def _load_ground_truth_and_create_mapping(self, dataset_path: str) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
        """
        Load ground truth data, create mapping from sequential index to UUID.
        """
        uuid_to_data_dict: Dict[str, Dict[str, Any]] = {}
        sequential_to_uuid_map: Dict[str, str] = {}
        potential_id_cols = ['id', 'sample_id', 'example_id', 'question_id']
        potential_answer_cols = ['answer', 'solution', 'target', 'completion']
        loaded_items = [] # Store items in loaded order

        try:
            # Attempt to load using datasets library first
            if is_datasets_available():
                try:
                    logger.info_rank0(f"Attempting to load {dataset_path} as HuggingFace dataset.")
                    loaded_hf_dataset = None
                    for file_type in ["json", "jsonl"]:
                        for config_name in ["train", None]:
                            try:
                                dataset = load_dataset(file_type, data_files=dataset_path, split=config_name, verification_mode=VerificationMode.NO_CHECKS)
                                logger.info_rank0(f"Successfully loaded as HuggingFace dataset ({file_type}, split: {config_name}).")
                                loaded_hf_dataset = dataset
                                break
                            except Exception:
                                continue
                        if loaded_hf_dataset: break
                    if loaded_hf_dataset:
                        dataset = loaded_hf_dataset
                        id_col_found = next((col for col in potential_id_cols if col in dataset.column_names), None)
                        answer_col_found = next((col for col in potential_answer_cols if col in dataset.column_names), None)

                        if id_col_found is None: raise ValueError("Missing ID column")
                        if answer_col_found is None: raise ValueError("Missing Answer column")

                        for i, item in enumerate(dataset):
                            item_dict = dict(item) # Convert to dict
                            raw_id = str(item_dict.get(id_col_found, f"hf_gen_{i}"))
                            norm_uuid = normalize_uuid(raw_id)
                            item_dict['original_id_col'] = id_col_found # Store original column name if needed
                            item_dict['original_id'] = raw_id
                            item_dict['normalized_uuid'] = norm_uuid
                            # Ensure answer is stored under 'answer' key
                            if answer_col_found != 'answer':
                                 item_dict['answer'] = item_dict.get(answer_col_found)
                            loaded_items.append(item_dict)

                except Exception as e:
                    logger.warning_rank0(f"Failed loading {dataset_path} as HuggingFace dataset: {e}. Falling back to manual JSON/JSONL.")

            # Fallback to manual loading if HF failed or items list is empty
            if not loaded_items:
                logger.info_rank0(f"Loading {dataset_path} as plain JSON/JSONL.")
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    try: data = json.load(f)
                    except json.JSONDecodeError:
                        f.seek(0); data = [json.loads(line) for line in f if line.strip()]

                if isinstance(data, list):
                    for i, item in enumerate(data):
                        if isinstance(item, dict): loaded_items.append(item)
                elif isinstance(data, dict): # If dict, keys are IDs
                    for i, (k, v) in enumerate(data.items()):
                        if isinstance(v, dict): v['id'] = k; loaded_items.append(v) # Add key as 'id' field
                else: raise ValueError("Loaded JSON is not list or dict.")

            # Process loaded items to create mappings
            missing_id_count = 0
            missing_answer_count = 0
            for i, item in enumerate(loaded_items):
                 # Find/Assign ID
                 item_id_raw = None
                 id_col_found = item.get('original_id_col') # Check if HF loading found it
                 if id_col_found and id_col_found in item:
                      item_id_raw = str(item[id_col_found])
                 else: # Manual search
                      for id_col in potential_id_cols:
                           if id_col in item: item_id_raw = str(item[id_col]); break
                 if item_id_raw is None:
                      item_id_raw = f"generated_idx_{i}"; missing_id_count += 1; item['id'] = item_id_raw

                 norm_uuid = normalize_uuid(item_id_raw)
                 sequential_id = str(i) # Key for the sequential mapping

                 # Find/Assign Answer
                 item_answer = None
                 if 'answer' in item: item_answer = item['answer']
                 else:
                      for ans_col in potential_answer_cols:
                           if ans_col in item: item_answer = item[ans_col]; item['answer'] = item_answer; break
                 if item_answer is None: missing_answer_count += 1; item['answer'] = None

                 # Populate dictionaries
                 if norm_uuid in uuid_to_data_dict: logger.warning_rank0(f"Duplicate normalized UUID {norm_uuid} found.")
                 uuid_to_data_dict[norm_uuid] = item
                 sequential_to_uuid_map[sequential_id] = norm_uuid

            if missing_id_count > 0: logger.warning_rank0(f"{missing_id_count} items missing ID, generated index-based IDs.")
            if missing_answer_count > 0: logger.warning_rank0(f"{missing_answer_count} items missing answer field.")

        except FileNotFoundError: logger.error_rank0(f"GT file not found: {dataset_path}"); raise
        except Exception as e: logger.error_rank0(f"Error loading GT data: {e}"); raise

        if not uuid_to_data_dict: logger.error_rank0("GT data dictionary is empty.")
        return uuid_to_data_dict, sequential_to_uuid_map

    def get_rewards(self, sample_ids: List[str], responses: List[str]) -> List[torch.Tensor]:
        """ Calculate rewards, using sequential to UUID mapping if needed. """
        rewards = []
        # Log first few IDs for debugging (only once)
        if not self._debug_logged and len(sample_ids) > 0:
             logger.info_rank0(f"Sample IDs received by get_rewards (first 5 raw): {sample_ids[:5]}")

        for i, (sample_id, response) in enumerate(zip(sample_ids, responses)):
            str_sample_id = str(sample_id) # Ensure string
            reward = self._calculate_single_reward(str_sample_id, response)
            rewards.append(torch.tensor([reward], dtype=torch.float))
        return rewards

    def _calculate_single_reward(self, sample_id: str, response: str) -> float:
        """ Calculate reward for a single response using ID mapping. """
        # --- Log IDs for debugging (only once per instance) ---
        if not self._debug_logged:
            self._debug_logged = True
            gt_keys_sample = list(self.uuid_to_data.keys())[:5]
            seq_keys_sample = list(self.sequential_to_uuid.items())[:5]
            logger.info_rank0(f"GT UUID keys sample: {gt_keys_sample}")
            logger.info_rank0(f"Sequential->UUID map sample: {seq_keys_sample}")
            logger.info_rank0(f"First sample_id received: '{sample_id}'")

        # --- Find corresponding UUID ---
        target_uuid = None
        # Try direct lookup (if ID is already a normalized UUID)
        norm_id = normalize_uuid(sample_id)
        if norm_id in self.uuid_to_data:
            target_uuid = norm_id
            if not self._debug_logged: logger.info_rank0(f"ID '{sample_id}' directly matched normalized UUID key.")
        # Else, try sequential mapping
        elif sample_id in self.sequential_to_uuid:
            target_uuid = self.sequential_to_uuid[sample_id]
            if not self._debug_logged: logger.info_rank0(f"ID '{sample_id}' mapped to UUID '{target_uuid}' via sequential map.")
        else:
            # Log specific failure case
            logger.warning_rank0(f"Sample ID '{sample_id}' (normalized: '{norm_id}') not found directly in GT keys or in sequential map.")
            return self.incorrect_reward

        # --- Retrieve GT data using the found UUID ---
        if target_uuid not in self.uuid_to_data:
             # This should ideally not happen if mapping is correct
             logger.error_rank0(f"Mapped/Normalized UUID '{target_uuid}' not found in gt_data dictionary keys!")
             return self.incorrect_reward

        gt_item = self.uuid_to_data[target_uuid]
        gt_answer = gt_item.get('answer')

        if gt_answer is None:
             logger.warning_rank0(f"Ground truth item found for UUID {target_uuid} (from ID {sample_id}), but 'answer' field is missing.")
             return self.incorrect_reward

        # --- Extract, Compare, and Check Format ---
        response_str = response.strip() # Use the full response string for format check
        # Use the updated extract_boxed_answer which handles both \boxed{ and boxed{
        boxed_answer_content = extract_boxed_answer(response_str)

        # Log comparison inputs regardless of extraction success
        log_extracted = boxed_answer_content if boxed_answer_content is not None else response_str # Log original if no box
        logger.info_rank0(f"Comparing for ID {sample_id} (UUID {target_uuid}):")
        logger.info_rank0(f"  Extracted/Raw Answer (from model response): {repr(log_extracted)} (Type: {type(log_extracted)})")
        logger.info_rank0(f"  Ground Truth Answer (from file): {repr(str(gt_answer))} (Type: {type(gt_answer)})")


        if boxed_answer_content is not None:
            is_correct = self._compare_answers(boxed_answer_content, str(gt_answer))
            # Check if the response is concise (only the boxed answer, with or without backslash)
            is_concise_strict = response_str == f"\\boxed{{{boxed_answer_content}}}"
            is_concise_no_bs = response_str == f"boxed{{{boxed_answer_content}}}"
            is_concise = is_concise_strict or is_concise_no_bs

            if is_correct:
                if is_concise:
                    reward = self.correct_reward # Correct answer and concise format (accepts missing \)
                    logger.debug(f"ID {sample_id}: Correct and Concise. Reward: {reward}")
                else:
                    # Correct answer but includes extra text (reasoning)
                    reward = self.correct_reward * 0.5 # Penalized positive reward for extra text
                    logger.info_rank0(f"ID {sample_id}: Correct but Not Concise (extra text). Penalized Reward: {reward}")
            else:
                reward = self.incorrect_reward # Incorrect answer
                logger.debug(f"ID {sample_id}: Incorrect Answer (Boxed Content Found). Reward: {reward}")
        else:
            # No boxed answer found (neither \boxed{} nor boxed{})
            logger.warning_rank0(f"ID {sample_id}: No boxed answer found in response. Reward: {self.incorrect_reward}")
            reward = self.incorrect_reward

        return reward

    # _extract_answer method is removed as its logic is integrated above or handled by extract_boxed_answer

    def _compare_answers(self, extracted_answer: str, ground_truth_answer: str) -> bool:
        """ Compares answers using mathematical equivalence checking. """
        # Ensure inputs are strings before passing to math_equal
        return math_equal(str(extracted_answer), str(ground_truth_answer), self.normalize)
