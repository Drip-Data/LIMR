import json
import numpy as np
from pathlib import Path
import argparse
from collections import defaultdict
import glob
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns # Optional: for nicer plots

# --- Add UUID Normalization Function (if needed, depends on original_prompts_path format) ---
# Assuming IDs in original_prompts_path and sample_metadata match directly or are handled upstream.
# def normalize_uuid(id_str):
#     """规范化UUID格式，确保一致性"""
#     if not id_str or not isinstance(id_str, str):
#         return str(id_str) if id_str is not None else "" # Return empty string for None
#     # 移除可能的空格和转为小写
#     return id_str.strip().lower()

def load_samples_llamafactory(train_samples_path, steps_per_epoch, max_epochs):
    """
    Loads sample data from LLaMA-Factory's jsonl output format.
    Handles cases where sample_metadata might be a dict or a list (for robustness).

    Args:
        train_samples_path (Path): Path to the directory containing samples_step_*.jsonl files.
        steps_per_epoch (int): Number of steps considered as one epoch.
        max_epochs (int): Maximum number of epochs to process.

    Returns:
        tuple: (epochs_data, all_sample_ids)
               epochs_data: A list of lists, where each inner list contains sample dicts for one epoch.
                            Sample dict format: {'original_id': str, 'reward': float, 'global_step': int}
               all_sample_ids: A set of all unique original_ids encountered.
    """
    epochs_data = []
    all_sample_ids = set()
    all_steps_data = defaultdict(list)
    max_step = steps_per_epoch * max_epochs

    # Find all sample files matching the pattern
    sample_files = sorted(
        glob.glob(os.path.join(train_samples_path, "samples_step_*.jsonl")),
        key=lambda x: int(Path(x).stem.split('_')[-1]) # Sort by step number
    )

    print(f"Found {len(sample_files)} sample files in {train_samples_path}")

    processed_steps = set()

    for file_path in sample_files:
        try:
            step_num_str = Path(file_path).stem.split('_')[-1]
            if not step_num_str.isdigit():
                print(f"Skipping file with non-numeric step: {file_path}")
                continue
            global_step = int(step_num_str)

            # Limit processing up to the calculated max_step based on max_epochs
            # Allow processing the step that completes the last epoch
            if global_step > max_step and (global_step -1) // steps_per_epoch >= max_epochs :
                 print(f"Skipping step {global_step} as it is beyond max_epochs {max_epochs}")
                 continue

            if global_step in processed_steps:
                print(f"Warning: Duplicate step file found for step {global_step}. Skipping {file_path}")
                continue
            processed_steps.add(global_step)

            # print(f"Processing step {global_step} from {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    try:
                        # Each line is a JSON object containing batch info
                        batch_data = json.loads(line.strip())
                        samples_to_process = []

                        # --- Corrected Logic to handle sample_metadata ---
                        if "sample_metadata" in batch_data:
                            if isinstance(batch_data["sample_metadata"], list):
                                # Handle case where it's a list (original assumption)
                                samples_to_process = batch_data["sample_metadata"]
                            elif isinstance(batch_data["sample_metadata"], dict):
                                # Handle case where it's a single dictionary
                                samples_to_process = [batch_data["sample_metadata"]]
                            else:
                                print(f"Warning: 'sample_metadata' is not a list or dict in line {line_num+1}, file {file_path}")

                        # Fallback check if sample_metadata is missing or invalid
                        elif "original_id" in batch_data and "reward" in batch_data:
                             print(f"Warning: Using top-level 'original_id' and 'reward' from line {line_num+1}, file {file_path} (older format?).")
                             # Adapt top-level data to look like sample_meta
                             samples_to_process = [{
                                 "original_id": batch_data.get("original_id"),
                                 "reward_value": batch_data.get("reward")
                             }]
                        # else:
                        #      print(f"Warning: No usable sample data found in line {line_num+1}, file {file_path}")

                        # Process the extracted sample(s)
                        for sample_meta in samples_to_process:
                            original_id = sample_meta.get("original_id")
                            # Use reward_value, default to incorrect if missing
                            reward_val = sample_meta.get("reward_value")
                            if reward_val is None:
                                # Maybe the key is just 'reward' inside sample_metadata?
                                reward_val = sample_meta.get("reward", -1.0) # Default to incorrect

                            try:
                                reward = float(reward_val)
                            except (ValueError, TypeError):
                                print(f"Warning: Could not convert reward '{reward_val}' to float for ID {original_id}. Using -1.0.")
                                reward = -1.0

                            if original_id is not None:
                                all_sample_ids.add(str(original_id))
                                # Store data associated with the step it was saved in
                                all_steps_data[global_step].append({
                                    'original_id': str(original_id),
                                    'reward': reward,
                                    'global_step': global_step
                                })
                            else:
                                 print(f"Warning: Missing 'original_id' in sample metadata, line {line_num+1}, file {file_path}")

                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON on line {line_num+1} in {file_path}: {e}")
                    except Exception as e:
                         print(f"Error processing line {line_num+1} in {file_path}: {e}")

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    # Group steps into epochs
    current_epoch_samples = []
    last_epoch_num = -1
    # Use max_step derived from max_epochs for iteration limit
    effective_max_step = steps_per_epoch * max_epochs
    sorted_steps = sorted(all_steps_data.keys())

    if not sorted_steps:
        print("Warning: No valid step data found.")
        return [], set()

    print(f"Processing {len(sorted_steps)} steps up to effective max_step {effective_max_step}")

    # Iterate based on steps_per_epoch up to max_epochs
    for epoch_idx in range(max_epochs):
        epoch_start_step = epoch_idx * steps_per_epoch + 1
        epoch_end_step = (epoch_idx + 1) * steps_per_epoch
        epoch_samples = []
        for step in range(epoch_start_step, epoch_end_step + 1):
            if step in all_steps_data:
                epoch_samples.extend(all_steps_data[step])

        if epoch_samples:
            print(f"Epoch {epoch_idx}: Found {len(epoch_samples)} samples between step {epoch_start_step} and {epoch_end_step}.")
            epochs_data.append(epoch_samples)
        else:
            print(f"Epoch {epoch_idx}: No samples found between step {epoch_start_step} and {epoch_end_step}.")
            epochs_data.append([]) # Append empty list to maintain structure

    return epochs_data, all_sample_ids


def calculate_reward_stats(epoch_data, all_sample_ids):
    """
    Calculates average reward for each sample ID within an epoch.

    Args:
        epoch_data (list): List of sample dicts for one epoch.
        all_sample_ids (set): Set of all unique original_ids.

    Returns:
        dict: {original_id: average_reward} for the epoch. Returns -1 if sample not in epoch.
    """
    rewards = {sample_id: [] for sample_id in all_sample_ids}

    for sample in epoch_data:
        sample_id = sample['original_id']
        if sample_id in rewards: # Should always be true if all_sample_ids is correct
             rewards[sample_id].append(sample['reward'])

    avg_rewards = {}
    for sample_id, reward_list in rewards.items():
        if reward_list:
            avg_rewards[sample_id] = np.mean(reward_list)
        else:
            avg_rewards[sample_id] = -1 # Indicate sample not present in this epoch

    return avg_rewards

def process_reward_sequences(sample_rewards_over_epochs, max_epochs):
    """
    Processes reward sequences, fills missing values, filters, and calculates baseline.

    Args:
        sample_rewards_over_epochs (dict): {original_id: [epoch0_avg_reward, epoch1_avg_reward, ...]}.
        max_epochs (int): Number of epochs to consider.

    Returns:
        tuple: (valid_ids, valid_sequences, baseline_sequence)
    """
    processed_sequences = {}
    # Ensure all sequences have length max_epochs, padding with -1
    for sample_id, sequence in sample_rewards_over_epochs.items():
        padded_sequence = sequence[:max_epochs] + [-1] * (max_epochs - len(sequence))
        processed_sequences[sample_id] = padded_sequence

    # Forward fill missing values (-1)
    for reward_sequence in processed_sequences.values():
        last_valid_reward = -1 # Start with invalid
        for i in range(max_epochs):
             if reward_sequence[i] != -1:
                  last_valid_reward = reward_sequence[i]
             elif last_valid_reward != -1: # Only fill if we have seen a valid reward before
                  reward_sequence[i] = last_valid_reward
        # print(f"FFilled sequence: {reward_sequence}") # Debug

    # Filter sequences that still have -1 after forward fill
    valid_sequences_data = {}
    for sample_id, sequence in processed_sequences.items():
        if -1 not in sequence:
            valid_sequences_data[sample_id] = sequence
        # else:
            # print(f"Filtering out sequence for {sample_id} due to missing values: {sequence}") # Debug

    if not valid_sequences_data:
        print("Warning: No valid reward sequences found after filtering.")
        return [], [], []

    valid_ids = list(valid_sequences_data.keys())
    valid_sequences = list(valid_sequences_data.values())

    if not valid_sequences: # Double check after potential filtering
         print("Warning: No valid sequences left after creating list.")
         return [], [], []

    # Calculate baseline (average reward sequence across valid samples)
    try:
        baseline_sequence = np.mean(valid_sequences, axis=0)
        print(f"Calculated baseline sequence (avg reward per epoch): {[f'{b:.3f}' for b in baseline_sequence]}")
    except Exception as e:
        print(f"Error calculating baseline sequence: {e}")
        print(f"Valid sequences sample: {valid_sequences[:2]}") # Print sample for debugging
        return [], [], []


    return valid_ids, valid_sequences, baseline_sequence

def calculate_similarity_score(sequence, baseline_sequence):
    """Calculates similarity based on the LIMR paper's formula."""
    if len(sequence) != len(baseline_sequence):
        print(f"Warning: Sequence length mismatch. Seq: {len(sequence)}, Baseline: {len(baseline_sequence)}")
        return 0.0 # Cannot calculate similarity

    # Use reward directly instead of accuracy (assuming reward reflects correctness/utility)
    # The formula compares the sequence to the baseline
    squared_diff_sum = sum((reward - baseline)**2 for reward, baseline in zip(sequence, baseline_sequence))

    # The denominator represents the maximum possible squared difference from the baseline
    # Assuming max possible reward is 1.0 (or self.correct_reward)
    # Let's use 1.0 as the theoretical max reward for the denominator calculation
    max_reward = 1.0
    max_diff_sum = sum((max_reward - baseline)**2 for baseline in baseline_sequence)

    if max_diff_sum < 1e-6: # Avoid division by zero if baseline is always max_reward
        return 1.0 if squared_diff_sum < 1e-6 else 0.0

    similarity = 1.0 - (squared_diff_sum / max_diff_sum)
    return max(0.0, similarity) # Ensure score is not negative

# --- Plotting Functions ---

def plot_average_reward(baseline_sequence, output_dir):
    """Plots the average reward over epochs."""
    if not baseline_sequence.size:
        print("Skipping average reward plot: baseline sequence is empty.")
        return
    epochs = range(len(baseline_sequence))
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, baseline_sequence, marker='o', linestyle='-')
    plt.title('Average Reward Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Average Reward')
    plt.grid(True)
    plt.xticks(epochs) # Ensure integer ticks for epochs
    plot_path = output_dir / "average_reward_curve.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved average reward curve to {plot_path}")

def plot_sample_rewards(sample_rewards_over_epochs, valid_ids, baseline_sequence, num_samples_to_plot, output_dir):
    """Plots reward sequences for selected samples against the baseline."""
    if not baseline_sequence.size or not valid_ids:
        print("Skipping sample reward plot: baseline or valid_ids is empty.")
        return

    epochs = range(len(baseline_sequence))
    plt.figure(figsize=(12, 7))

    # Plot baseline
    plt.plot(epochs, baseline_sequence, marker='o', linestyle='-', color='black', linewidth=2, label='Average Reward (Baseline)')

    # Select samples to plot
    if len(valid_ids) <= num_samples_to_plot:
        sample_ids_to_plot = valid_ids
    else:
        sample_ids_to_plot = random.sample(valid_ids, num_samples_to_plot)

    # Plot individual sample sequences
    for sample_id in sample_ids_to_plot:
        sequence = sample_rewards_over_epochs.get(sample_id)
        if sequence and len(sequence) == len(baseline_sequence):
             plt.plot(epochs, sequence, marker='.', linestyle='--', alpha=0.7, label=f'Sample {sample_id[:8]}...') # Shorten ID for legend

    plt.title(f'Reward Curves for {len(sample_ids_to_plot)} Samples vs. Average')
    plt.xlabel('Epoch')
    plt.ylabel('Average Reward')
    plt.grid(True)
    plt.xticks(epochs)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5)) # Place legend outside plot
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
    plot_path = output_dir / "sample_reward_curves.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved sample reward curves to {plot_path}")

def plot_similarity_distribution(sample_scores, output_dir):
    """Plots the distribution of similarity scores."""
    if not sample_scores:
        print("Skipping similarity distribution plot: no scores available.")
        return
    scores = list(sample_scores.values())
    plt.figure(figsize=(10, 6))
    sns.histplot(scores, kde=True, bins=20)
    plt.title('Distribution of Similarity Scores')
    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    plt.grid(True)
    plot_path = output_dir / "similarity_score_distribution.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved similarity score distribution to {plot_path}")


# --- Main Execution ---

def parse_args():
    parser = argparse.ArgumentParser(description='Process LLaMA-Factory PPO samples and filter prompts based on LIMR-like analysis.')
    parser.add_argument('--train_samples_path', type=str, required=True,
                        help='Path to the directory containing samples_step_*.jsonl files')
    parser.add_argument('--original_prompts_path', type=str, required=True,
                        help='Path to the original prompts json/jsonl file (must contain "id" and "prompt" fields)')
    parser.add_argument('--output_path', type=str, default='limr_filtered_prompts.jsonl',
                        help='Path for output filtered data (will be saved in jsonl format)')
    parser.add_argument('--plot_dir', type=str, default='./limr_plots',
                        help='Directory to save generated plots') # Added argument for plot directory
    parser.add_argument('--steps_per_epoch', type=int, default=80, # Adjusted default based on previous discussion
                        help='Number of steps that constitute one epoch for analysis')
    parser.add_argument('--max_epochs', type=int, default=20, # Adjusted default
                        help='Maximum number of epochs to consider for analysis')
    parser.add_argument('--similarity_threshold', type=float, default=0.2, # Default from original script
                        help='Minimum similarity score threshold for selecting prompts')
    parser.add_argument('--num_plot_samples', type=int, default=10,
                        help='Number of individual sample curves to plot') # Added argument for number of samples to plot
    return parser.parse_args()

def main():
    args = parse_args()
    train_samples_path = Path(args.train_samples_path)
    original_prompts_path = Path(args.original_prompts_path)
    output_path = Path(args.output_path)
    plot_dir = Path(args.plot_dir)

    # Create plot directory if it doesn't exist
    plot_dir.mkdir(parents=True, exist_ok=True)

    print("Loading saved samples...")
    # Load data using the adapted function
    epochs_data, all_sample_ids = load_samples_llamafactory(
        train_samples_path,
        args.steps_per_epoch,
        args.max_epochs
    )

    if not epochs_data or not all_sample_ids:
        print("No data loaded or no sample IDs found. Exiting.")
        return

    print(f"Loaded data for {len(epochs_data)} epochs. Found {len(all_sample_ids)} unique sample IDs.")

    print("Calculating average rewards per epoch for each sample ID...")
    # Calculate average reward per sample ID for each epoch
    epoch_avg_rewards = [calculate_reward_stats(epoch, all_sample_ids) for epoch in epochs_data]

    # Collect reward sequences for each sample ID
    sample_rewards_over_epochs = defaultdict(list)
    for epoch_rewards in epoch_avg_rewards:
        for sample_id, avg_reward in epoch_rewards.items():
            sample_rewards_over_epochs[sample_id].append(avg_reward)

    print("Processing reward sequences (filling missing, filtering, calculating baseline)...")
    # Process sequences (forward fill, filter, get baseline)
    valid_ids, reward_sequences, baseline_sequence = process_reward_sequences(
        sample_rewards_over_epochs, args.max_epochs)

    if not valid_ids:
        print("No valid samples left after processing sequences. Exiting.")
        return

    print(f"Number of valid samples with complete reward sequences: {len(valid_ids)}")

    print("Calculating similarity scores...")
    # Calculate similarity scores
    sample_scores = {}
    for sample_id, sequence in zip(valid_ids, reward_sequences):
         score = calculate_similarity_score(sequence, baseline_sequence)
         sample_scores[sample_id] = score
         # print(f"ID: {sample_id}, Score: {score:.4f}, Sequence: {[f'{s:.2f}' for s in sequence]}") # Debug

    # --- Generate Plots ---
    print("Generating plots...")
    plot_average_reward(baseline_sequence, plot_dir)
    plot_sample_rewards(sample_rewards_over_epochs, valid_ids, baseline_sequence, args.num_plot_samples, plot_dir)
    plot_similarity_distribution(sample_scores, plot_dir)
    print("Plots generated.")

    # Select prompts based on threshold
    selected_sample_ids = {sample_id for sample_id, score in sample_scores.items()
                           if score >= args.similarity_threshold}

    print(f"Selecting {len(selected_sample_ids)} samples with similarity score >= {args.similarity_threshold}")

    # Load original prompts and filter
    print(f"Loading original prompts from {original_prompts_path}...")
    filtered_data = []
    original_id_to_prompt = {} # Store original prompts for potential later use if needed
    try:
        with open(original_prompts_path, 'r', encoding='utf-8') as f:
            # Handle both json and jsonl for original prompts
            if original_prompts_path.suffix == '.json':
                original_data = json.load(f)
                if not isinstance(original_data, list):
                     print(f"Error: Expected a list of objects in {original_prompts_path}")
                     return
                for sample in original_data:
                    sample_id = str(sample.get('id')) # Assuming 'id' field exists
                    original_id_to_prompt[sample_id] = sample # Store the whole sample dict
                    if sample_id in selected_sample_ids:
                        filtered_data.append(sample)
            elif original_prompts_path.suffix == '.jsonl':
                 for line in f:
                      try:
                           sample = json.loads(line.strip())
                           sample_id = str(sample.get('id'))
                           original_id_to_prompt[sample_id] = sample
                           if sample_id in selected_sample_ids:
                                filtered_data.append(sample)
                      except json.JSONDecodeError:
                           print(f"Skipping invalid JSON line in {original_prompts_path}: {line.strip()}")
            else:
                 print(f"Error: Unsupported file type for original prompts: {original_prompts_path.suffix}")
                 return

    except FileNotFoundError:
        print(f"Error: Original prompts file not found at {original_prompts_path}")
        return
    except Exception as e:
        print(f"Error reading original prompts file: {e}")
        return

    print(f"Filtered down to {len(filtered_data)} samples.")

    # Save filtered data as jsonl
    print(f"Saving filtered data to {output_path}...")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in filtered_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print("Filtered data saved successfully.")
    except Exception as e:
        print(f"Error saving filtered data: {e}")

if __name__ == "__main__":
    main()
