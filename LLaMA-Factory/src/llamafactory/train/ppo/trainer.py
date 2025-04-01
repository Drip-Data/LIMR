# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's TRL library.
# https://github.com/huggingface/trl/blob/v0.8.0/trl/trainer/ppo_trainer.py
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

import math
import os
import sys
import warnings
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, List # Added List

import torch
from accelerate.utils import DistributedDataParallelKwargs
import torch # Ensure torch is imported for Generator
from torch.utils.data import DataLoader # Import DataLoader
from transformers.utils import is_datasets_available # Import is_datasets_available
from datasets import Dataset # Import Dataset

# Filter the specific deprecation warning early
warnings.filterwarnings("ignore", message="Trainer.tokenizer is now deprecated.*", category=DeprecationWarning)
# Filter other known noisy warnings
warnings.filterwarnings("ignore", message="You're using a Qwen2TokenizerFast tokenizer.*", category=UserWarning)
warnings.filterwarnings("ignore", message="No dataset is provided.*", category=UserWarning) # Related to PPOTrainer base class init
# Filter the RuleBasedRewardModel warning if needed during debugging (keep commented out unless needed)
# warnings.filterwarnings("ignore", message="Sample ID .* not found in ground truth dataset.", category=UserWarning)


from tqdm import tqdm
from transformers import GenerationConfig, Trainer, TrainerControl, TrainerState
from transformers.optimization import get_scheduler
from transformers.trainer import DEFAULT_CALLBACKS
from transformers.trainer_callback import CallbackHandler
from transformers.trainer_pt_utils import remove_dummy_checkpoint
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.utils import SAFE_WEIGHTS_NAME, WEIGHTS_NAME
from trl import PPOConfig, PPOTrainer
from trl.core import PPODecorators, logprobs_from_logits
from trl.models.utils import unwrap_model_for_generation
from typing_extensions import override

from ...extras import logging
from ...extras.misc import AverageMeter, count_parameters, get_current_device, get_logits_processor
from ..callbacks import FixValueHeadModelCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler
# Import normalize_uuid from reward module where it should be defined now
from ...reward import normalize_uuid
from .ppo_utils import dump_layernorm, get_rewards_from_server, replace_model, restore_layernorm


if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import (
        DataCollatorWithPadding,
        PreTrainedTokenizer,
        ProcessorMixin,
        Seq2SeqTrainingArguments,
        TrainerCallback,
    )
    from trl import AutoModelForCausalLMWithValueHead

    from ...hparams import FinetuningArguments, GeneratingArguments, ModelArguments
    # Import reward model types for type hinting
    from ...reward import RuleBasedRewardModel, RemoteRewardClient


logger = logging.get_logger(__name__)


# --- Prepare Dataset Function (Handles Pre-Tokenized Data, Ensures 'id' Column) ---
def prepare_dataset_for_ppo(dataset, tokenizer):
    """准备数据集用于PPO训练，处理已预分词的数据，并确保 'id' 列存在且保留。"""
    if dataset is None:
        return None

    logger.info_rank0("Preparing dataset for PPO training...")
    dataset_cols = set(dataset.column_names)

    # 检查核心分词列是否存在 (仅检查 input_ids，attention_mask 由 collator 生成)
    if not {'input_ids'}.issubset(dataset_cols):
        logger.error("Dataset provided to CustomPPOTrainer is missing 'input_ids'.")
        logger.error(f"Available columns: {list(dataset_cols)}")
        raise ValueError("PPO Trainer expects a dataset with 'input_ids'. Please check the data loading/processing pipeline.")

    # logger.info_rank0("Dataset appears to contain 'input_ids'.") # Adjusted log message

    # --- ID 列处理 (Revised Logic) ---
    id_col_to_use = 'id' # Target column name for the trainer
    id_column_preserved = None # Track which original column is used/preserved as 'id'

    # 1. Check if target 'id' column already exists and seems valid (e.g., contains UUIDs)
    if id_col_to_use in dataset_cols:
        # Simple check: assume if 'id' exists, it's the intended one (might need more robust check)
        logger.info_rank0(f"Found existing column named '{id_col_to_use}'. Assuming it contains correct IDs.")
        id_column_preserved = id_col_to_use
    else:
        # 2. If 'id' not found, search for other potential ID columns
        potential_id_columns = ['sample_id', 'example_id', 'question_id', 'uuid', 'original_id'] # Add more candidates
        found_potential_col = None
        for col in potential_id_columns:
            if col in dataset_cols:
                found_potential_col = col
                logger.info_rank0(f"Found potential ID column: '{found_potential_col}'.")
                break

        # 3. If a potential column is found, rename it to 'id'
        if found_potential_col:
            logger.info_rank0(f"Renaming column '{found_potential_col}' to '{id_col_to_use}'.")
            try:
                # No need to check if 'id' exists here, as we already established it doesn't in step 1
                dataset = dataset.rename_column(found_potential_col, id_col_to_use)
                dataset_cols.remove(found_potential_col)
                dataset_cols.add(id_col_to_use)
                id_column_preserved = id_col_to_use # Mark 'id' as preserved
            except Exception as e:
                 logger.error(f"Failed to rename ID column '{found_potential_col}' to '{id_col_to_use}': {e}. PPO might fail.")
                 # If rename fails, proceed without a guaranteed ID column

    # 4. If still no 'id' column after checks and potential rename, generate sequential IDs as last resort
    if id_column_preserved is None:
        logger.warning_rank0(f"No suitable existing ID column found or renaming failed. Adding sequential IDs as '{id_col_to_use}'. This might cause issues with reward calculation if ground truth uses different IDs.")
        if is_datasets_available() and isinstance(dataset, Dataset):
            # Ensure we don't overwrite if 'id' somehow exists now but wasn't marked preserved
            if id_col_to_use in dataset_cols:
                 logger.warning_rank0(f"Column '{id_col_to_use}' exists unexpectedly. Removing before adding sequential IDs.")
                 dataset = dataset.remove_columns([id_col_to_use])
                 dataset_cols.remove(id_col_to_use)
            dataset = dataset.add_column(id_col_to_use, [str(i) for i in range(len(dataset))])
            dataset_cols.add(id_col_to_use)
            id_column_preserved = id_col_to_use # Mark 'id' as preserved (even though sequential)
        else:
            logger.error(f"Cannot add sequential '{id_col_to_use}' column to non-standard dataset type.")
            # This is problematic, PPO will likely fail without IDs

    # --- 列移除 ---
    # 确定最终需要保留的列 (只保留 input_ids 和 ID 列)
    columns_to_keep = {'input_ids', id_column_preserved} # Removed attention_mask here
    columns_to_remove = list(dataset_cols - columns_to_keep)

    # 如果有需要移除的列，则移除
    if columns_to_remove:
        logger.info_rank0(f"Removing unnecessary columns for PPO input: {columns_to_remove}")
        try:
            processed_dataset = dataset.remove_columns(columns_to_remove)
        except ValueError as e:
             logger.error(f"Error removing columns: {e}. Existing columns: {dataset.column_names}")
             processed_dataset = dataset # Proceed with original dataset if removal fails
    else:
        processed_dataset = dataset

    # 设置最终格式 (只包含 input_ids 和 id)
    final_columns_list = [col for col in ['input_ids', 'id'] if col in processed_dataset.column_names] # Removed attention_mask
    try:
        processed_dataset.set_format(type="torch", columns=final_columns_list)
        logger.info_rank0(f"Dataset prepared for PPO Trainer init with columns: {final_columns_list}")
    except Exception as e:
         logger.error(f"Failed to set dataset format: {e}")
         raise # Re-raise the exception as this is critical

    return processed_dataset

# --- End Prepare Dataset Function ---


class CustomPPOTrainer(PPOTrainer, Trainer):
    r"""Inherit PPOTrainer."""

    def __init__(
        self,
        model_args: "ModelArguments",
        training_args: "Seq2SeqTrainingArguments",
        finetuning_args: "FinetuningArguments",
        generating_args: "GeneratingArguments",
        callbacks: Optional[list["TrainerCallback"]],
        model: "AutoModelForCausalLMWithValueHead",
        reward_model: Optional["AutoModelForCausalLMWithValueHead"],
        ref_model: Optional["AutoModelForCausalLMWithValueHead"],
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["ProcessorMixin"],
        data_collator: "DataCollatorWithPadding", # Should be DataCollatorWithPaddingAndIDs instance
        train_dataset: Optional["Dataset"] = None,
        eval_dataset: Optional["Dataset"] = None,
    ) -> None:
        if eval_dataset is not None:
            raise NotImplementedError("PPOTrainer does not support eval dataset yet.")

        # --- Prepare Dataset ---
        # The dataset should already be tokenized by the time it gets here
        # Pass tokenizer just in case the prepare function needs it (e.g., for future logic)
        processed_train_dataset = prepare_dataset_for_ppo(train_dataset, tokenizer)

        # --- PPO Config ---
        # Set DataLoader batch size based on user feedback/logs
        # Use per_device_train_batch_size from training_args for DataLoader
        data_loader_batch_size = training_args.per_device_train_batch_size
        # PPO mini_batch_size should match DataLoader batch size for this setup
        ppo_mini_batch_size = data_loader_batch_size
        gradient_accum_steps = training_args.gradient_accumulation_steps

        # Calculate the product required for divisibility check
        product = ppo_mini_batch_size * gradient_accum_steps

        # Calculate the desired buffer size based on finetuning args
        # Ensure ppo_buffer_size is at least 1
        buffer_size_multiplier = max(1, finetuning_args.ppo_buffer_size)
        desired_ppo_batch_size = ppo_mini_batch_size * buffer_size_multiplier

        # Adjust ppo_batch_size to be the nearest multiple of 'product' >= desired_ppo_batch_size
        buffer_multiplier_factor = math.ceil(desired_ppo_batch_size / product)
        adjusted_ppo_batch_size = product * buffer_multiplier_factor

        logger.info_rank0(
            f"DataLoader batch size (per device): {data_loader_batch_size}, "
            f"PPO mini_batch_size: {ppo_mini_batch_size}, grad_accum_steps: {gradient_accum_steps}. "
            f"Calculated product for adjustment: {product}. "
            f"Original ppo_buffer_size multiplier: {buffer_size_multiplier}. Desired PPO batch_size: {desired_ppo_batch_size}. "
            f"Adjusting PPO batch_size (buffer size) to {adjusted_ppo_batch_size} (multiple of {product})."
        )


        ppo_config = PPOConfig(
            model_name=model_args.model_name_or_path,
            learning_rate=training_args.learning_rate,
            mini_batch_size=ppo_mini_batch_size,
            batch_size=adjusted_ppo_batch_size, # Use the adjusted size
            gradient_accumulation_steps=gradient_accum_steps,
            ppo_epochs=finetuning_args.ppo_epochs,
            max_grad_norm=training_args.max_grad_norm,
            seed=training_args.seed,
            optimize_device_cache=True,
            target=finetuning_args.ppo_target,
            use_score_scaling=finetuning_args.ppo_score_norm,
            use_score_norm=finetuning_args.ppo_score_norm,
            whiten_rewards=finetuning_args.ppo_whiten_rewards,
            accelerator_kwargs={"step_scheduler_with_optimizer": False},
            log_with=training_args.report_to[0] if training_args.report_to else None,
            project_kwargs={"logging_dir": training_args.logging_dir},
            remove_unused_columns=False, # Keep this False
        )

        # Add deepspeed config
        if training_args.deepspeed_plugin is not None:
            ppo_config.accelerator_kwargs["kwargs_handlers"] = [
                DistributedDataParallelKwargs(find_unused_parameters=training_args.ddp_find_unused_parameters)
            ]
            ppo_config.accelerator_kwargs["deepspeed_plugin"] = training_args.deepspeed_plugin
            if ppo_config.log_with is not None:
                logger.warning_rank0("PPOTrainer cannot use external logger when DeepSpeed is enabled.")
                ppo_config.log_with = None

        # --- Optimizer and Scheduler ---
        # Calculate num_training_steps based on the *processed* dataset
        if processed_train_dataset is not None:
            if training_args.max_steps > 0:
                num_training_steps = training_args.max_steps
            else:
                # Calculate total steps based on processed dataset length and adjusted batch sizes
                # Use the adjusted_ppo_batch_size calculated earlier
                # Number of PPO steps (updates) per epoch
                num_update_steps_per_epoch = math.ceil(len(processed_train_dataset) / (adjusted_ppo_batch_size * training_args.world_size))
                num_training_steps = int(training_args.num_train_epochs * num_update_steps_per_epoch) # Ensure int
        else:
            num_training_steps = 0 # Or handle error if no dataset

        optimizer = self.create_optimizer(model, training_args, finetuning_args)
        scheduler = self.create_scheduler(training_args, num_training_steps, optimizer)

        # --- Initialize PPOTrainer (base class) ---
        # Pass the adjusted ppo_config
        # Pass dataset=None to prevent internal DataLoader creation
        PPOTrainer.__init__(
            self,
            config=ppo_config,
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            dataset=None, # IMPORTANT: Keep as None
            optimizer=optimizer,
            data_collator=data_collator, # Use the provided collator
            lr_scheduler=scheduler,
        )

        # --- Ensure consistent padding for decoder-only models ---
        self.tokenizer.padding_side = "left"
        logger.info_rank0(f"Set tokenizer padding_side to 'left'.")


        # --- Create Custom DataLoader ---
        if processed_train_dataset is not None:
            data_loader_kwargs = {
                "batch_size": data_loader_batch_size, # Use per_device size here
                "collate_fn": data_collator, # Use the provided collator
                "num_workers": training_args.dataloader_num_workers,
                "pin_memory": training_args.dataloader_pin_memory,
                "drop_last": False, # Keep False for PPO
                "shuffle": True # Shuffle training data
            }
            if training_args.seed is not None:
                 data_loader_kwargs["generator"] = torch.Generator().manual_seed(training_args.seed)

            self.dataloader = DataLoader(processed_train_dataset, **data_loader_kwargs)
            # Assign the processed dataset to self.dataset as well
            self.dataset = processed_train_dataset
        else:
            self.dataloader = None
            self.dataset = None

        # --- Remaining Initialization ---
        self.args = training_args
        self.model_args = model_args
        self.finetuning_args = finetuning_args
        self.reward_model = reward_model
        self.current_device = get_current_device()

        self.generation_config = GenerationConfig(
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=[self.tokenizer.eos_token_id] + self.tokenizer.additional_special_tokens_ids,
            **generating_args.to_dict(),
        )

        self.state = TrainerState()
        self.control = TrainerControl()
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        callbacks = DEFAULT_CALLBACKS if callbacks is None else DEFAULT_CALLBACKS + callbacks
        self.callback_handler = CallbackHandler(
            callbacks, self.accelerator.unwrap_model(self.model), self.tokenizer, self.optimizer, self.lr_scheduler
        )
        if self.args.max_steps > 0:
            logger.info_rank0("max_steps is given, it will override any value given in num_train_epochs")

        self.amp_context = torch.autocast(self.current_device.type)
        warnings.simplefilter("ignore")

        if finetuning_args.reward_model_type == "full":
            if self.is_deepspeed_enabled:
                if not (
                    getattr(reward_model.pretrained_model, "is_loaded_in_8bit", False)
                    or getattr(reward_model.pretrained_model, "is_loaded_in_4bit", False)
                ):
                    self.reward_model = self._prepare_deepspeed(self.reward_model)
            else:
                self.reward_model = self.accelerator.prepare_model(self.reward_model, evaluation_mode=True)

            self.add_callback(FixValueHeadModelCallback)
        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        # Initialize cache for ID mapping in _get_limr_rewards
        self._id_mapping_cache = {}

        # --- Initialize SampleWriter for LIMR if enabled ---
        self._limr_sample_writer = None
        self._batch_counter = 0 # Moved counter here
        self._total_samples_saved = 0 # Moved counter here
        if hasattr(self.args, 'limr') and self.args.limr.enabled and self.args.limr.save_samples_path is not None: # Check limr args exist
            if self.is_world_process_zero(): # Only rank 0 initializes the writer
                from ...data.sample_writer import SampleWriter
                import datetime # Import datetime here as well

                # Ensure save path directory exists
                save_dir = os.path.dirname(self.args.limr.save_samples_path)
                if save_dir: # Check if save_dir is not empty
                     os.makedirs(save_dir, exist_ok=True)
                     logger.info_rank0(f"Ensured directory exists: {save_dir}")
                else:
                     logger.warning_rank0(f"Sample save path seems to be just a filename: {self.args.limr.save_samples_path}. Saving to current directory.")
                     save_dir = "." # Default to current directory if only filename given

                # Adjust save path to be just the directory for the writer if needed
                # Assuming save_samples_path is the base path/prefix, not the full filename pattern
                self._limr_sample_writer = SampleWriter(
                    save_path=save_dir, # Pass the directory path
                    distributed=self.args.world_size > 1,
                    rank=self.args.local_process_index
                )
                logger.info_rank0(f"Initialized SampleWriter in __init__ for rank 0. Save directory: {save_dir}")


        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version
            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)


    # --- Simplified ppo_train method based on user feedback ---
    def ppo_train(self, resume_from_checkpoint: Optional[str] = None) -> None:
        if resume_from_checkpoint is not None:
            raise ValueError("`resume_from_checkpoint` is not supported in this custom loop yet.")

        # Ensure dataloader and dataset are initialized
        if self.dataloader is None or self.dataset is None:
            raise ValueError("DataLoader or Dataset not initialized. Provide `train_dataset`.")

        # Calculate total training steps
        if self.args.max_steps > 0:
            max_steps = self.args.max_steps
            num_examples = self.config.batch_size * self.args.gradient_accumulation_steps * max_steps # Approx
            num_train_epochs = sys.maxsize
            steps_in_epoch = 0 # Avoid division by zero if max_steps is used
        else:
            # Calculate steps based on dataloader length and PPO buffer size
            # Number of batches needed to fill the buffer once
            batches_per_buffer = math.ceil(self.config.batch_size / self.config.mini_batch_size)
            # Number of PPO steps per epoch
            num_ppo_steps_per_epoch = math.ceil(len(self.dataloader) / batches_per_buffer)
            max_steps = math.ceil(self.args.num_train_epochs * num_ppo_steps_per_epoch)
            num_examples = len(self.dataset)
            num_train_epochs = self.args.num_train_epochs
            steps_in_epoch = num_ppo_steps_per_epoch # PPO steps per epoch

        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        logger.info_rank0("***** Running PPO Training (Simplified Loop) *****")
        logger.info_rank0(f"  Num examples = {num_examples:,}")
        logger.info_rank0(f"  Num Epochs = {num_train_epochs:,}")
        logger.info_rank0(f"  DataLoader batch size = {self.config.mini_batch_size:,}") # Should match dataloader
        logger.info_rank0(f"  PPO buffer size (config.batch_size) = {self.config.batch_size:,}")
        logger.info_rank0(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps:,}") # Note: grad accum handled by PPOTrainer.step
        logger.info_rank0(f"  Num optimization epochs per batch = {self.config.ppo_epochs:,}")
        logger.info_rank0(f"  Total PPO training steps = {max_steps:,}")

        dataiter = iter(self.dataloader)
        loss_meter = AverageMeter()
        reward_meter = AverageMeter()
        self.callback_handler.on_train_begin(self.args, self.state, self.control)

        # --- Main Training Loop ---
        for step in tqdm(range(max_steps), disable=not self.is_local_process_zero()):
            experience_buffer = {
                "queries": [],
                "responses": [],
                "rewards": [],
                "sample_ids": []
            }
            buffer_target_size = self.config.batch_size

            # --- Experience Gathering Phase ---
            self.model.eval()
            self.tokenizer.padding_side = "right"

            collected_count = 0
            while collected_count < buffer_target_size:
                try:
                    batch = next(dataiter)
                except StopIteration:
                    dataiter = iter(self.dataloader)
                    batch = next(dataiter)

                # Validate batch content (already preprocessed)
                if not all(k in batch and batch[k] is not None for k in ["input_ids", "attention_mask", "id"]):
                    logger.warning(f"Skipping batch at PPO step {step} due to missing keys.")
                    continue
                if not isinstance(batch["input_ids"], torch.Tensor) or batch["input_ids"].numel() == 0:
                    logger.warning(f"Skipping batch at PPO step {step} due to invalid 'input_ids'.")
                    continue

                # Generate responses
                # Pass batch directly as it's already a dict of tensors
                mini_batch_queries, mini_batch_responses, mini_batch_ids = self.get_inputs(batch)

                # Process only valid generations
                if mini_batch_queries and len(mini_batch_queries) > 0:
                    mini_batch_rewards = self.get_rewards(mini_batch_queries, mini_batch_responses, mini_batch_ids)

                    # Ensure rewards are valid tensors and match count
                    valid_rewards = [r for r in mini_batch_rewards if isinstance(r, torch.Tensor)]
                    if len(valid_rewards) == len(mini_batch_ids):
                        # Add to experience buffer (move tensors to CPU)
                        experience_buffer["queries"].extend([q.cpu() for q in mini_batch_queries])
                        experience_buffer["responses"].extend([r.cpu() for r in mini_batch_responses])
                        experience_buffer["rewards"].extend([r.cpu() for r in valid_rewards])
                        experience_buffer["sample_ids"].extend(mini_batch_ids)
                        collected_count += len(mini_batch_ids)
                    else:
                         logger.warning(f"Reward mismatch/failure for batch at PPO step {step}. Skipping.")
                else:
                    logger.warning(f"Generation failed for batch at PPO step {step}. Skipping.")

            # --- PPO Optimization Phase ---
            if collected_count < buffer_target_size:
                logger.warning(f"Collected only {collected_count}/{buffer_target_size} experiences for PPO step {step}. Skipping update.")
                self.state.global_step += 1 # Increment step even if skipped
                continue # Skip PPO update

            # Trim buffer exactly to buffer_target_size
            for key in experience_buffer:
                experience_buffer[key] = experience_buffer[key][:buffer_target_size]

            # Prepare PPO update data (move to device)
            query_tensors = [q.to(self.current_device) for q in experience_buffer["queries"]]
            response_tensors = [r.to(self.current_device) for r in experience_buffer["responses"]]
            reward_tensors = [r.to(self.current_device) for r in experience_buffer["rewards"]]

            # Execute PPO update
            self.model.train()
            # self.tokenizer.padding_side = "left" # Already set in __init__

            # --- Restructured Try/Except for Step and Saving ---
            step_succeeded = False
            stats = None
            try:
                stats = self.step(query_tensors, response_tensors, reward_tensors)
                step_succeeded = True
            except Exception as e:
                 # Log error specifically from the step
                 logger.error(f"Error during PPO step {step}: {e}", exc_info=True) # Add traceback

            # Always update global step (whether step succeeded or not)
            self.state.global_step += 1

            # --- Save LIMR Samples (Moved outside the try block) ---
            if (self.is_world_process_zero() and # Ensure only rank 0 saves
                self._limr_sample_writer is not None and # Check if writer was initialized
                hasattr(self.args, 'limr') and self.args.limr.enabled and # Check limr args exist
                self.args.limr.save_samples_path is not None and
                hasattr(self.state, 'global_step') and # Should exist now
                self.state.global_step > 0 and # Avoid saving at step 0
                self.state.global_step % self.args.limr.save_every_n_steps == 0):

                import time # Import time and datetime here if not already imported globally
                import datetime

                logger.info_rank0(f"Attempting to save {len(experience_buffer['sample_ids'])} samples for completed step {self.state.global_step}")

                # Decode queries and responses (which are on CPU in experience_buffer)
                try:
                    decoded_queries = self.tokenizer.batch_decode(
                        [q.cpu() for q in experience_buffer["queries"]], # Ensure CPU
                        skip_special_tokens=True
                    )
                    decoded_responses = self.tokenizer.batch_decode(
                        [r.cpu() for r in experience_buffer["responses"]], # Ensure CPU
                        skip_special_tokens=True
                    )
                    # Rewards are already tensors on CPU in experience_buffer

                    # Preprocess rewards before passing to ensure no None values are passed directly
                    valid_rewards_for_saving = []
                    for reward in experience_buffer["rewards"]:
                        if reward is not None:
                            valid_rewards_for_saving.append(reward)
                        else:
                            # Replace None with a zero tensor on the correct device (CPU in this case)
                            logger.warning_rank0(f"Found None reward before saving at step {self.state.global_step}. Replacing with 0.0 tensor.")
                            valid_rewards_for_saving.append(torch.tensor([0.0], device='cpu')) # Ensure it's a tensor on CPU

                    # Call the _save_limr_samples method
                    self._save_limr_samples(
                        experience_buffer["sample_ids"],
                        decoded_queries,
                        decoded_responses,
                        valid_rewards_for_saving # Pass the processed list
                    )
                except Exception as save_e:
                     logger.error(f"Error during sample decoding or saving at step {self.state.global_step}: {save_e}", exc_info=True)


            # Log metrics only if step succeeded
            if step_succeeded and stats is not None:
                loss_meter.update(float(stats["ppo/loss/total"]), n=buffer_target_size)
                # Ensure reward_tensors is not empty before stacking
                if reward_tensors:
                     reward_meter.update(torch.stack(reward_tensors).mean().item(), n=buffer_target_size) # Use reward_tensors from device
                else:
                     logger.warning(f"No valid reward tensors to log at step {self.state.global_step}")


                self.callback_handler.on_step_end(self.args, self.state, self.control)

                # Logging logic
                if self.is_local_process_zero() and self.state.global_step % self.args.logging_steps == 0:
                    logs = dict(
                        loss=round(loss_meter.avg, 4),
                        reward=round(reward_meter.avg, 4),
                        learning_rate=stats["ppo/learning_rate"],
                        epoch=round(self.state.global_step / steps_in_epoch, 2) if steps_in_epoch > 0 else 0, # Use global_step
                    )
                    tqdm.write(str(logs))
                    logs["step"] = self.state.global_step # Log global step
                    self.state.log_history.append(logs)
                    self.callback_handler.on_log(self.args, self.state, self.control, logs)
                    loss_meter.reset()
                    reward_meter.reset()
            elif not step_succeeded:
                 # Optionally log that metrics are skipped due to step failure
                 logger.warning(f"Skipping metric logging at step {self.state.global_step} due to PPO step failure.")


            # Save checkpoint logic (runs regardless of step success)
            if self.state.global_step % self.args.save_steps == 0:
                self.save_model(
                    os.path.join(self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}")
                )
                self.callback_handler.on_save(self.args, self.state, self.control)

            if self.control.should_epoch_stop or self.control.should_training_stop:
                break

        self.callback_handler.on_train_end(self.args, self.state, self.control)


    # --- Keep remaining methods (create_optimizer, create_scheduler, get_inputs, get_rewards, etc.) as they were ---
    # --- They should work correctly with the preprocessed data and buffer ---

    @override
    def create_optimizer(
        self,
        model: "AutoModelForCausalLMWithValueHead",
        training_args: "Seq2SeqTrainingArguments",
        finetuning_args: "FinetuningArguments",
    ) -> "torch.optim.Optimizer":
        optimizer = create_custom_optimizer(model, training_args, finetuning_args)
        if optimizer is None:
            decay_params, nodecay_params = [], []
            decay_param_names = self.get_decay_parameter_names(model)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if name in decay_param_names:
                        decay_params.append(param)
                    else:
                        nodecay_params.append(param)

            optim_class, optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args)

            # --- Explicitly convert learning rate to float ---
            if "lr" in optim_kwargs:
                try:
                    optim_kwargs["lr"] = float(optim_kwargs["lr"])
                except (ValueError, TypeError) as e:
                    logger.error(f"Failed to convert learning rate '{optim_kwargs['lr']}' to float: {e}. Using default.")
                    # Fallback or re-raise depending on desired behavior
                    optim_kwargs["lr"] = 1e-5 # Example fallback

            param_groups = [
                dict(params=nodecay_params),
                dict(params=decay_params, weight_decay=training_args.weight_decay),
            ]
            optimizer = optim_class(param_groups, **optim_kwargs)

        return optimizer

    @override
    def create_scheduler(
        self, training_args: "Seq2SeqTrainingArguments", num_training_steps: int, optimizer: "torch.optim.Optimizer"
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(training_args, num_training_steps, optimizer)
        lr_scheduler = get_scheduler(
            training_args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=training_args.get_warmup_steps(num_training_steps),
            num_training_steps=num_training_steps,
        )
        return lr_scheduler

    @torch.no_grad()
    def get_inputs(self, batch: dict[str, Any]) -> tuple[list["torch.Tensor"], list["torch.Tensor"], list[str]]:
        r"""Generate model's responses given queries (input_ids from batch)
           and return corresponding sample IDs."""
        # Extract sample IDs, handle missing 'id' key or None value
        if "id" not in batch or batch["id"] is None:
            logger.error("Sample IDs ('id') not found or is None in the batch.")
            # Determine batch size from input_ids if possible, else default to 0
            num_samples = len(batch.get("input_ids", []))
            sample_ids = [f"error_id_{i}" for i in range(num_samples)] # Generate placeholder IDs
        else:
            # Ensure IDs are strings
            sample_ids = [str(sid) for sid in batch["id"]]

        # Check for required keys for generation
        required_gen_keys = ["input_ids", "attention_mask"]
        if not all(key in batch and batch[key] is not None for key in required_gen_keys):
             logger.error(f"Missing required keys for generation in batch: {required_gen_keys}. Batch keys: {batch.keys()}")
             # Return empty lists and original/placeholder IDs
             return [], [], sample_ids

        # Prepare batch for generation: use input_ids and attention_mask directly
        # No need to remove 'id' as generate doesn't use it.
        # Move tensors to the correct device
        gen_kwargs = {}
        for key in required_gen_keys:
            if isinstance(batch[key], torch.Tensor):
                gen_kwargs[key] = batch[key].to(self.current_device)
            else:
                # Handle cases where expected tensors might not be tensors (e.g., lists)
                # This might indicate an upstream issue, but attempt conversion if possible
                try:
                    gen_kwargs[key] = torch.tensor(batch[key]).to(self.current_device)
                    logger.warning(f"Converted batch key '{key}' to tensor. Original type: {type(batch[key])}")
                except (TypeError, ValueError) as e:
                    logger.error(f"Failed to convert batch key '{key}' to tensor: {e}. Skipping generation.")
                    return [], [], sample_ids

        # Add other potential generation arguments if they exist in the batch and are tensors
        for key, value in batch.items():
            if key not in required_gen_keys and key != "id" and isinstance(value, torch.Tensor):
                gen_kwargs[key] = value.to(self.current_device)


        # Generation logic (remains largely the same)
        with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
            unwrapped_model: AutoModelForCausalLMWithValueHead = self.accelerator.unwrap_model(self.model)
            if self.model_args.upcast_layernorm:
                layernorm_params = dump_layernorm(unwrapped_model)

            try:
                generate_output: torch.Tensor = unwrapped_model.generate(
                    generation_config=self.generation_config, logits_processor=get_logits_processor(), **gen_kwargs
                )
            except Exception as e:
                 logger.error(f"Error during model generation: {e}. Generation kwargs keys: {gen_kwargs.keys()}")
                 return [], [], sample_ids # Return original IDs on error

            if self.model_args.upcast_layernorm:
                restore_layernorm(unwrapped_model, layernorm_params)

        # Queries are the input_ids from the batch (move back to CPU)
        queries_tensor = batch["input_ids"].detach().cpu()
        # Responses are the generated part (move back to CPU)
        # Ensure generate_output is valid before slicing
        if generate_output is None or not isinstance(generate_output, torch.Tensor) or generate_output.dim() < 2:
             logger.error("Invalid generation output received.")
             return [], [], sample_ids

        # Ensure query tensor has at least 2 dimensions for size comparison
        if queries_tensor.dim() < 2:
             logger.error(f"Invalid query tensor dimension: {queries_tensor.dim()}")
             return [], [], sample_ids

        responses_tensor = generate_output[:, queries_tensor.size(-1) :].detach().cpu()


        # Convert tensors back to lists of tensors, handling padding/EOS
        # especially when decoding for reward calculation.
        queries_list = [q for q in queries_tensor]
        responses_list = []
        for i in range(len(responses_tensor)):
             response_i = responses_tensor[i]
             # Find the end of the generated sequence (first EOS or pad token)
             # This logic might need adjustment based on tokenizer behavior
             non_pad_indices = (response_i != self.tokenizer.pad_token_id).nonzero(as_tuple=True)[0] # Use as_tuple=True
             if len(non_pad_indices) == 0:
                 response_length = 1 # Handle empty or all-padding response
             else:
                 # Check for EOS token if it's different from pad token
                 eos_indices = (response_i == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0] # Use as_tuple=True
                 if len(eos_indices) > 0 and self.tokenizer.eos_token_id != self.tokenizer.pad_token_id:
                      # Find the first EOS token index
                      first_eos_index = eos_indices[0].item()
                      response_length = first_eos_index + 1
                 else: # Use last non-pad token index + 1
                      response_length = non_pad_indices[-1].item() + 1

             responses_list.append(response_i[:response_length]) # Trim padding/EOS

        # Ensure lengths match before returning
        if len(queries_list) != len(responses_list) or len(queries_list) != len(sample_ids):
             logger.error(f"Mismatch after processing: Q({len(queries_list)}), R({len(responses_list)}), ID({len(sample_ids)}).")
             # Attempt to return based on the shortest list to avoid index errors
             min_len = min(len(queries_list), len(responses_list), len(sample_ids))
             return queries_list[:min_len], responses_list[:min_len], sample_ids[:min_len]

        return queries_list, responses_list, sample_ids

    @torch.no_grad()
    def get_rewards(
        self,
        queries: list["torch.Tensor"],
        responses: list["torch.Tensor"],
        sample_ids: list[str], # Added sample_ids parameter
    ) -> list["torch.Tensor"]:
        r"""Compute scores using given reward model.

        Both inputs and outputs are put on CPU.
        sample_ids are used for LIMR reward calculation.
        """
        # Check if LIMR is enabled and use appropriate reward mechanism
        if hasattr(self.args, 'limr') and self.args.limr.enabled: # Check if limr args exist
            # Pass sample_ids to _get_limr_rewards
            return self._get_limr_rewards(queries, responses, sample_ids)

        # Original reward calculation logic (remains unchanged for non-LIMR)
        # Ensure reward_model is available if LIMR is disabled
        if self.reward_model is None and self.finetuning_args.reward_model_type != "api":
             logger.error("Reward model is None, but required for non-LIMR PPO reward calculation.")
             # Return dummy rewards or raise error
             return [torch.tensor([0.0]) for _ in queries] # Example dummy reward


        if self.finetuning_args.reward_model_type == "api":
            token_ids = [torch.cat((q, r), dim=-1).tolist() for q, r in zip(queries, responses)]
            messages = self.tokenizer.batch_decode(token_ids, skip_special_tokens=False)
            return get_rewards_from_server(self.reward_model, messages)

        batch: dict[str, torch.Tensor] = self.prepare_model_inputs(queries, responses)
        unwrapped_model: AutoModelForCausalLMWithValueHead = self.accelerator.unwrap_model(self.model)

        if self.finetuning_args.reward_model_type == "lora":
            replace_model(unwrapped_model, target="reward")
            reward_model_instance = self.model # Use the main model with reward adapter
        else:
            reward_model_instance = self.reward_model # Use the separate reward model instance

        # Ensure reward_model_instance is not None before proceeding
        if reward_model_instance is None:
             logger.error("Reward model instance is None during reward calculation.")
             return [torch.tensor([0.0]) for _ in queries]

        with unwrap_model_for_generation(reward_model_instance, self.accelerator), self.amp_context:  # support bf16
            # Move batch to the correct device
            model_inputs_on_device = {k: v.to(self.current_device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            try:
                outputs = reward_model_instance(**model_inputs_on_device, return_dict=True, use_cache=False)
                # Assuming the reward/value is the last element in the output tuple or dict
                if isinstance(outputs, dict):
                    # Common case: value head output might be named 'rewards' or 'values'
                    values = outputs.get("rewards", outputs.get("values"))
                elif isinstance(outputs, (list, tuple)):
                    values = outputs[-1]
                else:
                    values = outputs # Assume direct value output if not dict/list/tuple

                if values is None or not isinstance(values, torch.Tensor):
                     logger.error(f"Could not extract reward/value tensor from reward model output. Output type: {type(outputs)}")
                     return [torch.tensor([0.0]) for _ in queries]

            except Exception as e:
                 logger.error(f"Error during reward model forward pass: {e}")
                 return [torch.tensor([0.0]) for _ in queries]


        if self.finetuning_args.reward_model_type == "lora":
            replace_model(unwrapped_model, target="default")

        # Ensure attention_mask is on the correct device for gather
        attention_mask_on_device = batch["attention_mask"].to(values.device)
        # Calculate the index for the last token's value
        last_token_indices = attention_mask_on_device.sum(dim=-1, keepdim=True) - 1
        # Ensure indices are within bounds
        last_token_indices = last_token_indices.clamp(min=0, max=values.size(1) - 1)
        rewards = values.gather(dim=1, index=last_token_indices) # Gather along seq_len dim
        return rewards.float().cpu().detach() # Move rewards back to CPU and detach

    def _get_limr_rewards(
        self,
        queries: list["torch.Tensor"],
        responses: list["torch.Tensor"],
        sample_ids: list[str], # Added sample_ids parameter
    ) -> list["torch.Tensor"]:
        r"""Compute rewards using LIMR reward mechanism with robust ID mapping."""
        from ...reward import RuleBasedRewardModel, RemoteRewardClient, normalize_uuid # Import normalize_uuid

        # Lazy initialization of LIMR components
        if self.args.limr.reward_type == "remote":
            if not hasattr(self, "_limr_reward_client") or self._limr_reward_client is None:
                if self.args.limr.reward_url is None:
                     raise ValueError("reward_url must be specified for remote LIMR reward.")
                self._limr_reward_client = RemoteRewardClient(self.args.limr.reward_url)
            reward_calculator = self._limr_reward_client # Assuming remote client handles ID mapping internally if needed
        elif self.args.limr.reward_type == "rule":
            if not hasattr(self, "_limr_reward_model") or self._limr_reward_model is None:
                if self.args.limr.ground_truth_dataset is None:
                    raise ValueError("ground_truth_dataset must be specified for rule-based LIMR reward.")
                # RuleBasedRewardModel now loads gt_data and sequential_to_uuid mapping
                self._limr_reward_model = RuleBasedRewardModel(
                    ground_truth_dataset=self.args.limr.ground_truth_dataset,
                    correct_reward=self.args.limr.reward_correct,
                    incorrect_reward=self.args.limr.reward_incorrect,
                    normalize=self.args.limr.math_equal_normalize
                )
            reward_calculator = self._limr_reward_model
        else:
             raise ValueError(f"Unsupported LIMR reward type: {self.args.limr.reward_type}")

        # Decode queries and responses
        decoded_queries = self.tokenizer.batch_decode(queries, skip_special_tokens=True)
        decoded_responses = self.tokenizer.batch_decode(responses, skip_special_tokens=True)

        # --- Enhanced ID Mapping Logic (for rule-based reward) ---
        if self.args.limr.reward_type == "rule":
            logger.info_rank0(f"_get_limr_rewards received sample_ids (first 5): {sample_ids[:5]}...")

            # Ensure reward_calculator and its necessary attributes are initialized
            if not hasattr(reward_calculator, 'uuid_to_data') or not hasattr(reward_calculator, 'sequential_to_uuid'):
                 logger.error_rank0("Reward calculator is missing 'uuid_to_data' or 'sequential_to_uuid' mapping. Cannot proceed.")
                 # Return incorrect rewards for the whole batch
                 # Use reward_calculator's incorrect_reward if available, else fallback
                 incorrect_reward_val = getattr(reward_calculator, 'incorrect_reward', -1.0)
                 return [torch.tensor([incorrect_reward_val]) for _ in sample_ids]


            mapped_sample_ids = []
            for sid in sample_ids:
                sid_str = str(sid)
                mapped_id = None

                # 1. Check cache first
                if sid_str in self._id_mapping_cache:
                    mapped_id = self._id_mapping_cache[sid_str]
                else:
                    # 2. Try direct UUID match (after normalization)
                    norm_id = normalize_uuid(sid_str)
                    if norm_id in reward_calculator.uuid_to_data:
                        mapped_id = norm_id
                        # logger.debug(f"Direct UUID match for ID '{sid_str}' (normalized: '{norm_id}')") # Debug log
                    # 3. Try sequential mapping lookup
                    elif sid_str in reward_calculator.sequential_to_uuid:
                        mapped_id = reward_calculator.sequential_to_uuid[sid_str] # This gives the normalized UUID
                        # logger.debug(f"Sequential map lookup for ID '{sid_str}' -> UUID '{mapped_id}'") # Debug log
                    # 4. Try index-based lookup (if sid_str is numeric)
                    else:
                        try:
                            idx = int(sid_str)
                            if 0 <= idx < len(reward_calculator.sequential_to_uuid):
                                 seq_key = str(idx)
                                 if seq_key in reward_calculator.sequential_to_uuid:
                                      mapped_id = reward_calculator.sequential_to_uuid[seq_key] # This gives the normalized UUID
                                      # logger.debug(f"Index-based lookup for ID '{sid_str}' -> SeqKey '{seq_key}' -> UUID '{mapped_id}'") # Debug log
                                 else:
                                      logger.warning_rank0(f"Index {idx} (from ID '{sid_str}') valid, but key '{seq_key}' not in sequential map.")
                            else:
                                 logger.warning_rank0(f"Index {idx} (from ID '{sid_str}') is out of bounds for sequential map (size {len(reward_calculator.sequential_to_uuid)}).")
                        except ValueError: pass # sid_str is not a simple integer
                        except Exception as e: logger.error_rank0(f"Unexpected error during index lookup for ID '{sid_str}': {e}")

                    # Update cache if a mapping was found
                    if mapped_id:
                        self._id_mapping_cache[sid_str] = mapped_id
                    else:
                        # Fallback: use original normalized ID (though it likely won't match GT keys)
                        mapped_id = norm_id # Use normalized original ID as fallback key
                        logger.warning_rank0(f"Failed to map ID '{sid_str}' (normalized: '{norm_id}'). Using normalized ID for reward lookup.")
                        self._id_mapping_cache[sid_str] = mapped_id # Cache the fallback

                mapped_sample_ids.append(mapped_id) # This list now contains the best-effort UUIDs/keys

            # Use the mapped IDs (which should be normalized UUIDs or fallbacks) for reward calculation
            rewards = reward_calculator.get_rewards(mapped_sample_ids, decoded_responses)

        elif self.args.limr.reward_type == "remote":
             # Assuming remote client handles its own ID logic
             rewards = reward_calculator.get_rewards(sample_ids, decoded_queries, decoded_responses)
        else:
             rewards = [torch.tensor([self.incorrect_reward]) for _ in sample_ids]


        # Ensure rewards are tensors
        processed_rewards = []
        for r in rewards:
             if not isinstance(r, torch.Tensor):
                  try:
                       # Attempt conversion, default to incorrect reward on failure
                       processed_rewards.append(torch.tensor([float(r)]))
                  except (ValueError, TypeError):
                       logger.warning(f"Failed to convert reward '{r}' to tensor. Using incorrect reward.")
                       processed_rewards.append(torch.tensor([self.args.limr.reward_incorrect]))
             else:
                  processed_rewards.append(r)

        # --- Sample saving moved to ppo_train loop ---

        return processed_rewards # Return list of tensors


    def _save_limr_samples( # Keep the function signature for now, but it won't be called from here
        self,
        sample_ids: list[str],
        queries: list[str],
        responses: list[str],
        rewards: list["torch.Tensor"],
    ) -> None:
        r"""Save samples with detailed metadata for reward tracking.
           This function is now called from the ppo_train loop after each step.
        """
        # from ...data.sample_writer import SampleWriter # Already imported in __init__
        import time
        import datetime

        # Check if writer exists (should have been initialized in __init__ for rank 0)
        if self._limr_sample_writer is None:
             logger.warning_rank0("SampleWriter not initialized. Skipping sample saving.")
             return

        # No need to check distributed/rank here, as this function is only called by rank 0

        # 添加详细日志
        logger.info_rank0(f"Attempting to save {len(sample_ids)} samples at step {self.state.global_step}")
        # logger.info_rank0(f"Save path (directory): {self._limr_sample_writer.save_path}") # Log directory path

        # Create unique batch identifier with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Use a counter specific to saving steps if needed, or just use global_step
        # self._batch_counter is less meaningful now, maybe reset it per save step?
        # Let's use global_step directly for simplicity in batch_id
        batch_id = f"step{self.state.global_step}_fullbuffer_{timestamp}"
        # self._batch_counter += 1 # Counter might not be needed or reset differently

        # Calculate reward statistics (handle None values)
        batch_size = len(sample_ids) # Use sample_ids length for consistency
        reward_values = []
        valid_rewards = []

        for reward in rewards:
            if reward is not None:
                # 确保reward是标量值
                try:
                    if hasattr(reward, 'item'):
                        reward_val = reward.item()  # 从张量中提取标量值
                    else:
                        reward_val = float(reward)  # 尝试转换为float
                    reward_values.append(reward_val)
                    valid_rewards.append(reward_val)
                except (ValueError, TypeError) as e:
                     logger.warning_rank0(f"Failed to convert reward '{reward}' to float in _save_limr_samples: {e}. Treating as None.")
                     reward_values.append(None) # Append None if conversion fails
            else:
                reward_values.append(None)  # 保持None值不变

        # 计算有效奖励的统计信息
        if valid_rewards:
            avg_reward = sum(valid_rewards) / len(valid_rewards)
            max_reward = max(valid_rewards)
            min_reward = min(valid_rewards)
        else: # Handle case where all rewards were None or invalid
            avg_reward = 0.0
            max_reward = 0.0
            min_reward = 0.0
            logger.warning_rank0("No valid rewards found in the buffer to calculate statistics.")

        # Create detailed per-sample metadata
        sample_metadata = []
        start_global_position = self._total_samples_saved
        for i, (sid, reward_val) in enumerate(zip(sample_ids, reward_values)): # reward_val can be None here
            # Calculate normalized reward only if value is valid and range exists
            reward_normalized = 0.0
            if reward_val is not None and (max_reward - min_reward) > 1e-5:
                 reward_normalized = (reward_val - min_reward) / (max_reward - min_reward)

            sample_metadata.append({
                "original_id": sid,
                "batch_id": batch_id,
                "position_in_buffer": i, # Position within this saved buffer
                "global_position": start_global_position + i, # Approximate global position
                "reward_value": reward_val, # Can be None
                "reward_normalized": reward_normalized, # Use calculated normalized value
                "timestamp": time.time(),
                "datetime": datetime.datetime.now().isoformat()
            })

        # Update total samples counter
        self._total_samples_saved += batch_size

        # Prepare batch metadata (ensure values are JSON serializable)
        batch_metadata = {
            "global_step": int(self.state.global_step) if hasattr(self.state, 'global_step') and self.state.global_step is not None else -1,
            "epoch": float(self.state.epoch) if hasattr(self.state, 'epoch') and self.state.epoch is not None else -1.0, # Check for None
            "batch_id": batch_id,
            "timestamp": time.time(),
            "datetime": datetime.datetime.now().isoformat(),
            "batch_size": batch_size,
            "avg_reward": avg_reward,
            "max_reward": max_reward,
            "min_reward": min_reward,
            "total_samples_saved_so_far": self._total_samples_saved, # Renamed for clarity
            "sample_metadata": sample_metadata # Pass the list of dicts
        }

        # Pass enhanced metadata to the writer
        # Ensure rewards passed are suitable for the writer (e.g., list of floats or tensors it handles)
        # Process rewards list to handle None values before passing
        processed_rewards_for_writer = [
            r.item() if r is not None and hasattr(r, 'item') else (float(r) if r is not None else 0.0)
            for r in rewards
        ]
        self._limr_sample_writer.write_batch(
            step=int(self.state.global_step) if hasattr(self.state, 'global_step') else -1, # Ensure step is int
            sample_ids=sample_ids,
            prompts=queries, # Already decoded strings
            responses=responses, # Already decoded strings
            rewards=processed_rewards_for_writer, # Pass the processed list with None replaced by 0.0
            metadata=batch_metadata # Pass the batch metadata dict
        )

        # Log summary of saved samples (already done in the calling loop)
        # logger.info_rank0(
        #     f"Saved {batch_size} samples at step {self.state.global_step}. " # Removed batch counter info
        #     f"Avg reward: {avg_reward:.4f}, Min: {min_reward:.4f}, Max: {max_reward:.4f}"
        # )


    @override
    @PPODecorators.empty_device_cache()
    def batched_forward_pass(
        self,
        model: "AutoModelForCausalLMWithValueHead",
        queries: "torch.Tensor",
        responses: "torch.Tensor",
        model_inputs: dict[str, Any],
        return_logits: bool = False,
        response_masks: Optional["torch.Tensor"] = None,
    ) -> tuple["torch.Tensor", Optional["torch.Tensor"], "torch.Tensor", "torch.Tensor"]:
        r"""Calculate model outputs in multiple batches.

        Subclass and override to inject custom behavior.
        """
        bs = len(queries)
        fbs = self.config.mini_batch_size
        all_logprobs = []
        all_logits = []
        all_masks = []
        all_values = []

        for i in range(math.ceil(bs / fbs)):
            input_kwargs = {key: value[i * fbs : (i + 1) * fbs] for key, value in model_inputs.items()}
            query_batch = queries[i * fbs : (i + 1) * fbs]
            response_batch = responses[i * fbs : (i + 1) * fbs]
            if response_masks is not None:
                response_masks_batch = response_masks[i * fbs : (i + 1) * fbs]
            input_ids = input_kwargs["input_ids"]
            attention_mask = input_kwargs["attention_mask"]

            # --- Add Logging ---
            logger.debug(f"batched_forward_pass - Batch {i}: input_ids shape: {input_ids.shape}, attention_mask shape: {attention_mask.shape}")


            with self.amp_context:  # support bf16
                logits, _, values = model(**input_kwargs, return_dict=True, use_cache=False)

            logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
            masks = torch.zeros_like(attention_mask)
            masks[:, :-1] = attention_mask[:, 1:]

            for j in range(len(query_batch)):
                # Calculate start and end indices for response tokens
                query_len = torch.sum(attention_mask[j, :len(query_batch[j])]).item() # Length of query based on attention mask
                start = query_len -1 # Index of the last query token

                # Handle potential left padding (find first non-pad token)
                if attention_mask[j, 0] == 0:
                    try:
                        # Find the index of the first '1' in the attention mask
                        first_non_pad = attention_mask[j, :].nonzero(as_tuple=True)[0][0].item()
                        start = first_non_pad + torch.sum(attention_mask[j, first_non_pad:len(query_batch[j])]).item() - 1
                    except IndexError: # Handle cases where the entire query might be padding (shouldn't happen ideally)
                         logger.warning(f"Could not find non-padding tokens in query {j} of batch.")
                         start = len(query_batch[j]) - 1 # Fallback

                response_len = len(response_batch[j])
                end = start + response_len

                # --- Add Logging ---
                logger.debug(f"batched_forward_pass - Batch {i}, Item {j}: query_len={query_len}, start={start}, response_len={response_len}, end={end}, input_ids_len={input_ids.shape[1]}")


                # Check for potential issues leading to empty mask
                if start < 0 or end <= start or end > masks.shape[1]:
                     logger.warning(f"batched_forward_pass - Batch {i}, Item {j}: Invalid mask indices! start={start}, end={end}, mask_shape={masks.shape}")

                # Apply mask: 0 for query tokens, 1 for response tokens
                # Ensure indices are valid before masking
                safe_start = max(0, start)
                safe_end = min(masks.shape[1], end)

                # Ensure start+1 is within bounds before masking query
                if safe_start + 1 < masks.shape[1]:
                    masks[j, :safe_start+1] = 0 # Mask query tokens (including the last one)
                elif safe_start + 1 == masks.shape[1]: # Handle case where query is the entire sequence length
                     masks[j, :] = 0
                     logger.warning(f"batched_forward_pass - Batch {i}, Item {j}: Query seems to fill the entire sequence length ({safe_start+1}). Masking all.")

                # Ensure end is within bounds before masking padding after response
                if safe_end < masks.shape[1]:
                    masks[j, safe_end:] = 0    # Mask padding tokens after response

                # Check if the resulting mask for the response part is all zeros AFTER applying query/padding masks
                if safe_end > safe_start + 1:
                    response_mask_segment = masks[j, safe_start+1:safe_end]
                    if torch.all(response_mask_segment == 0):
                         logger.warning(f"batched_forward_pass - Batch {i}, Item {j}: Response mask segment is all zeros! Indices: {safe_start+1} to {safe_end}")
                elif safe_end <= safe_start + 1: # Handle case where response length is 0 or negative
                     logger.warning(f"batched_forward_pass - Batch {i}, Item {j}: Calculated response length is zero or negative (start={start}, end={end}). Mask might be empty.")


                # Apply response_masks if provided (e.g., for ignoring padding within response)
                if response_masks is not None:
                    # Ensure response_masks_batch aligns with the response part of the combined sequence
                    # This part might need careful adjustment based on how response_masks are generated
                    # Assuming response_masks_batch corresponds directly to the response tokens
                    if safe_end > safe_start + 1: # Only apply if there's a valid response segment
                        response_mask_part = response_masks_batch[j][:response_len] # Get the relevant part
                        # Ensure the mask part has the correct length
                        expected_len = safe_end - (safe_start + 1)
                        if len(response_mask_part) == expected_len:
                             masks[j, safe_start+1:safe_end] = masks[j, safe_start+1:safe_end] * response_mask_part # Apply mask
                        else:
                             logger.warning(f"batched_forward_pass - Batch {i}, Item {j}: Mismatch between response_mask_part length ({len(response_mask_part)}) and expected length ({expected_len}). Skipping response_mask application.")

                # --- Add Mask Safety Check ---
                # Safety check - ensure mask has at least one non-zero element
                if masks[j].sum() == 0:
                    # Set at least one token (preferably the first response token if possible) to be attended
                    safe_idx = min(safe_start + 1, masks.size(1) - 1) # Use safe_start
                    if safe_idx >= 0: # Ensure index is valid
                        masks[j, safe_idx] = 1
                        logger.warning(f"Fixed empty mask for sample {j} in batch {i} by setting index {safe_idx} to 1.")
                    else:
                         logger.error(f"Could not fix empty mask for sample {j} in batch {i} - invalid safe_idx {safe_idx}")


            # --- Add Logging ---
            # Check if any mask in the batch is all zeros AFTER all masking operations
            if torch.any(torch.all(masks == 0, dim=1)):
                 zero_mask_indices = torch.where(torch.all(masks == 0, dim=1))[0]
                 logger.warning(f"batched_forward_pass - Batch {i}: Found all-zero mask(s) AFTER processing at indices: {zero_mask_indices.tolist()}")


            if return_logits:
                all_logits.append(logits)
            else:
                del logits

            all_values.append(values)
            all_logprobs.append(logprobs)
            all_masks.append(masks)

        # Final check before returning - ensure no all-zero masks are passed
        final_masks = torch.cat(all_masks)[:, :-1]
        if torch.any(torch.all(final_masks == 0, dim=1)):
             zero_final_mask_indices = torch.where(torch.all(final_masks == 0, dim=1))[0]
             logger.error(f"batched_forward_pass - Returning final masks containing all zeros at indices: {zero_final_mask_indices.tolist()}. This will likely cause IndexError in TRL.")
             # Potentially raise an error here or return dummy valid masks?
             # For now, just log the error.

        return (
            torch.cat(all_logprobs),
            torch.cat(all_logits)[:, :-1] if return_logits else None,
            torch.cat(all_values)[:, :-1],
            final_masks, # Return the potentially problematic masks
        )

    @override
    def save_model(self, output_dir: Optional[str] = None) -> None:
        r"""Save model checkpoint.

        Subclass and override to inject custom behavior.
        """
        if output_dir is None:
            output_dir = self.args.output_dir

        if self.is_fsdp_enabled or self.is_deepspeed_enabled:
            try:
                state_dict = self.accelerator.get_state_dict(self.model)  # must be called at all ranks
                if self.args.should_save:
                    self._save(output_dir, state_dict=state_dict)
            except ValueError:
                logger.warning_rank0(
                    " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead,"
                    " use zero_to_fp32.py to recover weights"
                )
                if self.args.should_save:
                    self._save(output_dir, state_dict={})
                # remove the dummy state_dict
                remove_dummy_checkpoint(self.args.should_save, output_dir, [WEIGHTS_NAME, SAFE_WEIGHTS_NAME])
                self.model.save_checkpoint(output_dir)

        elif self.args.should_save:
            unwrapped_model: AutoModelForCausalLMWithValueHead = self.accelerator.unwrap_model(self.model)
            self._save(output_dir, state_dict=unwrapped_model.state_dict())

