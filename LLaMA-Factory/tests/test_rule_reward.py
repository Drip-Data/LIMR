import sys
import os
import pytest
import torch

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    from llamafactory.reward.rule_reward import RuleBasedRewardModel
    from llamafactory.extras.math_utils import HAS_SYMPY
except ImportError as e:
    print(f"Error importing llamafactory modules: {e}")
    print(f"sys.path: {sys.path}")
    pytest.skip("Skipping tests due to import error.", allow_module_level=True)

# Point to the correct location of test_ground_truth.json (one level above project_root)
GROUND_TRUTH_PATH = os.path.join(project_root, "..", "data/test_ground_truth.json")

@pytest.fixture(scope="module")
def reward_model():
    # Ensure the ground truth file exists before creating the model
    if not os.path.exists(GROUND_TRUTH_PATH):
         pytest.skip(f"Ground truth file not found at {GROUND_TRUTH_PATH}", allow_module_level=True)
    return RuleBasedRewardModel(
        ground_truth_dataset=GROUND_TRUTH_PATH,
        correct_reward=1.5,
        incorrect_reward=-0.5,
        normalize=True # Default, test with normalization
    )

@pytest.fixture(scope="module")
def reward_model_no_norm():
     if not os.path.exists(GROUND_TRUTH_PATH):
         pytest.skip(f"Ground truth file not found at {GROUND_TRUTH_PATH}", allow_module_level=True)
     return RuleBasedRewardModel(
        ground_truth_dataset=GROUND_TRUTH_PATH,
        correct_reward=1.0,
        incorrect_reward=-1.0,
        normalize=False # Test without normalization
    )

def test_reward_calculation_correct(reward_model):
    sample_ids = ["id1", "id2"]
    responses = ["Response with \\boxed{ans_A}", "Another response \\boxed{ans_B}"]
    rewards = reward_model.get_rewards(sample_ids, responses)
    assert len(rewards) == 2
    assert torch.isclose(rewards[0], torch.tensor([1.5]))
    assert torch.isclose(rewards[1], torch.tensor([1.5]))

def test_reward_calculation_incorrect(reward_model):
    sample_ids = ["id1", "id2"]
    responses = ["Response with \\boxed{wrong_A}", "Another response \\boxed{wrong_B}"]
    rewards = reward_model.get_rewards(sample_ids, responses)
    assert len(rewards) == 2
    assert torch.isclose(rewards[0], torch.tensor([-0.5]))
    assert torch.isclose(rewards[1], torch.tensor([-0.5]))

def test_reward_calculation_missing_box(reward_model):
    sample_ids = ["id1"]
    responses = ["Response without box"]
    rewards = reward_model.get_rewards(sample_ids, responses)
    assert len(rewards) == 1
    assert torch.isclose(rewards[0], torch.tensor([-0.5]))

def test_reward_calculation_missing_id(reward_model):
    sample_ids = ["id_missing"]
    responses = ["Response \\boxed{ans_A}"]
    rewards = reward_model.get_rewards(sample_ids, responses)
    assert len(rewards) == 1
    assert torch.isclose(rewards[0], torch.tensor([-0.5])) # Should be incorrect if ID not found

def test_reward_calculation_mixed(reward_model):
    sample_ids = ["id1", "id2", "id_missing", "id1"]
    responses = [
        "Correct \\boxed{ans_A}",
        "Incorrect \\boxed{wrong_B}",
        "Missing ID \\boxed{ans_C}",
        "No box here"
    ]
    rewards = reward_model.get_rewards(sample_ids, responses)
    assert len(rewards) == 4
    assert torch.isclose(rewards[0], torch.tensor([1.5]))
    assert torch.isclose(rewards[1], torch.tensor([-0.5]))
    assert torch.isclose(rewards[2], torch.tensor([-0.5]))
    assert torch.isclose(rewards[3], torch.tensor([-0.5]))

@pytest.mark.skipif(not HAS_SYMPY, reason="sympy not installed")
def test_reward_calculation_math_correct_sympy(reward_model):
    sample_ids = ["id_math"]
    responses = ["The answer is \\boxed{y + 2*x}"] # Equivalent expression
    rewards = reward_model.get_rewards(sample_ids, responses)
    assert len(rewards) == 1
    assert torch.isclose(rewards[0], torch.tensor([1.5]))

@pytest.mark.skipif(not HAS_SYMPY, reason="sympy not installed")
def test_reward_calculation_math_incorrect_sympy(reward_model):
    sample_ids = ["id_math"]
    responses = ["The answer is \\boxed{2*x - y}"] # Incorrect expression
    rewards = reward_model.get_rewards(sample_ids, responses)
    assert len(rewards) == 1
    assert torch.isclose(rewards[0], torch.tensor([-0.5]))

def test_reward_calculation_no_norm(reward_model_no_norm):
    sample_ids = ["id_math"]
    # Exact match needed when normalize=False
    responses_correct = ["Answer: \\boxed{2*x + y}"]
    responses_incorrect = ["Answer: \\boxed{y + 2*x}"] # Incorrect due to order/spacing if not normalized

    rewards_correct = reward_model_no_norm.get_rewards(sample_ids, responses_correct)
    rewards_incorrect = reward_model_no_norm.get_rewards(sample_ids, responses_incorrect)

    assert torch.isclose(rewards_correct[0], torch.tensor([1.0]))
    assert torch.isclose(rewards_incorrect[0], torch.tensor([-1.0]))
