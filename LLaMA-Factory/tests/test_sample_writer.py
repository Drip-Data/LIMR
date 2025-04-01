import sys
import os
import json
import tempfile
import shutil
import pytest
from pytest import approx
import torch

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    from llamafactory.data.sample_writer import SampleWriter
except ImportError as e:
    print(f"Error importing llamafactory modules: {e}")
    print(f"sys.path: {sys.path}")
    pytest.skip("Skipping tests due to import error.", allow_module_level=True)


@pytest.fixture
def temp_save_dir():
    # Create a temporary directory for saving samples
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Clean up the temporary directory after tests
    shutil.rmtree(temp_dir)

def test_sample_writer_init(temp_save_dir):
    writer = SampleWriter(save_path=temp_save_dir)
    assert writer.save_path == temp_save_dir
    assert os.path.exists(temp_save_dir)

def test_write_batch_single_process(temp_save_dir):
    writer = SampleWriter(save_path=temp_save_dir, distributed=False, rank=0)
    step = 100
    sample_ids = ["id_a", "id_b"]
    prompts = ["prompt a", "prompt b"]
    responses = ["response a", "response b"]
    rewards = [torch.tensor([1.0]), torch.tensor([-1.0])] # Use tensors as in trainer
    metadata = {"extra_info": ["info_a", "info_b"]}

    writer.write_batch(step, sample_ids, prompts, responses, rewards, metadata)

    expected_filepath = os.path.join(temp_save_dir, f"samples_step_{step}.jsonl")
    assert os.path.exists(expected_filepath)

    lines = []
    with open(expected_filepath, "r", encoding="utf-8") as f:
        for line in f:
            lines.append(json.loads(line))

    assert len(lines) == 2
    assert lines[0] == {
        "id": "id_a",
        "prompt": "prompt a",
        "response": "response a",
        "reward": 1.0, # Should be float in JSON
        "extra_info": "info_a"
    }
    assert lines[1] == {
        "id": "id_b",
        "prompt": "prompt b",
        "response": "response b",
        "reward": -1.0,
        "extra_info": "info_b"
    }

def test_write_batch_distributed_rank_0(temp_save_dir):
    writer = SampleWriter(save_path=temp_save_dir, distributed=True, rank=0)
    step = 200
    sample_ids = ["id_c"]
    prompts = ["prompt c"]
    responses = ["response c"]
    rewards = [torch.tensor([0.5])]

    writer.write_batch(step, sample_ids, prompts, responses, rewards)

    expected_filepath = os.path.join(temp_save_dir, f"samples_step_{step}.jsonl")
    assert os.path.exists(expected_filepath) # Rank 0 should write

    with open(expected_filepath, "r", encoding="utf-8") as f:
        line = json.loads(f.readline())
    assert line["id"] == "id_c"

def test_write_batch_distributed_rank_non_0(temp_save_dir):
    writer = SampleWriter(save_path=temp_save_dir, distributed=True, rank=1) # Non-zero rank
    step = 300
    sample_ids = ["id_d"]
    prompts = ["prompt d"]
    responses = ["response d"]
    rewards = [torch.tensor([0.0])]

    writer.write_batch(step, sample_ids, prompts, responses, rewards)

    expected_filepath = os.path.join(temp_save_dir, f"samples_step_{step}.jsonl")
    assert not os.path.exists(expected_filepath) # Non-zero rank should not write

def test_write_sample(temp_save_dir):
    writer = SampleWriter(save_path=temp_save_dir)
    step = 50
    sample_id = "single_id"
    prompt = "single prompt"
    response = "single response"
    reward = torch.tensor([0.9])
    metadata = {"meta_key": "meta_value"}

    writer.write_sample(step, sample_id, prompt, response, reward, metadata)

    expected_filepath = os.path.join(temp_save_dir, f"samples_step_{step}.jsonl")
    assert os.path.exists(expected_filepath)

    with open(expected_filepath, "r", encoding="utf-8") as f:
        line = json.loads(f.readline())

    # Use pytest.approx for floating point comparison
    assert line["id"] == "single_id"
    assert line["prompt"] == "single prompt"
    assert line["response"] == "single response"
    assert line["reward"] == approx(0.9) 
    assert line["meta_key"] == "meta_value"
