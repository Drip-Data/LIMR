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

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Optional

from ...extras import logging
from ..data_utils import Role
from .processor_utils import DatasetProcessor, infer_seqlen


if TYPE_CHECKING:
    from ..mm_plugin import AudioInput, ImageInput, VideoInput


logger = logging.get_logger(__name__)


class UnsupervisedDatasetProcessor(DatasetProcessor):
    def _encode_data_example(
        self,
        prompt: list[dict[str, str]],
        response: list[dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
    ) -> list[int]: # Return only input_ids
        if len(response) == 1:
            messages = prompt + response
        else:
            messages = prompt + [{"role": Role.ASSISTANT.value, "content": ""}]

        # Process multimedia first
        messages = self.template.mm_plugin.process_messages(messages, images, videos, audios, self.processor)

        # Encode the messages to get input_ids (prompt part)
        # We don't need the labels returned by encode_oneturn for PPO's unsupervised phase
        input_ids, _ = self.template.encode_oneturn(self.tokenizer, messages, system, tools)

        # Apply cutoff length to the input_ids (prompt)
        # For PPO, we only care about the prompt length here.
        source_len, _ = infer_seqlen(len(input_ids), 0, self.data_args.cutoff_len)
        input_ids = input_ids[:source_len]

        return input_ids # Return only the processed input_ids

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # build inputs with format `<bos> X`
        model_inputs = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1:
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue

            input_ids = self._encode_data_example( # Now only returns input_ids
                prompt=examples["_prompt"][i],
                response=examples["_response"][i], # Still needed to determine message structure
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
            )

            if not input_ids: # Skip if encoding failed or returned empty
                 logger.warning_rank0(f"Skipping example index {i} due to empty input_ids after encoding.")
                 continue

            # Add only input_ids and id to model_inputs for PPO
            model_inputs["input_ids"].append(input_ids)
            # Attention mask and labels are NOT added here. Collator will handle mask.

            # --- ID Handling ---
            # Try to get the ID from the '_id' field added by the converter
            original_id = examples["_id"][i] if "_id" in examples and examples["_id"][i] is not None else None

            if original_id is not None:
                # Use the original ID (already stringified by converter)
                sample_id = str(original_id)
            else:
                # If _id was None or missing, generate sequential ID based on index 'i'
                # This assumes the order is preserved and matches the ground truth loading order
                logger.warning_rank0(f"Missing or None '_id' for example index {i}. Generating sequential ID '{i}'.")
                sample_id = str(i) # Use the index 'i' as the sequential ID

            # Ensure the output column is named 'id' (without underscore)
            model_inputs["id"].append(sample_id)

        return model_inputs

    # Restore and modify print_data_example to satisfy the abstract method requirement
    def print_data_example(self, example: dict[str, list[int]]) -> None:
        """Prints a data example for debugging, adapted for PPO unsupervised stage."""
        # Check if 'input_ids' exists before trying to access it
        if "input_ids" in example:
            print("input_ids:\n{}".format(example["input_ids"]))
            # Decode and print the inputs using the tokenizer
            print("inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        else:
            print("Example does not contain 'input_ids'.")
        # 'labels' are not processed in this stage, so they are not printed.
