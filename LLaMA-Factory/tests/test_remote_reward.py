import sys
import os
import pytest
import torch
from unittest import mock

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    from llamafactory.reward.remote_reward import RemoteRewardClient
    from llamafactory.extras.packages import is_requests_available
    if is_requests_available():
        import requests # Import only if available
except ImportError as e:
    print(f"Error importing llamafactory modules: {e}")
    print(f"sys.path: {sys.path}")
    pytest.skip("Skipping tests due to import error.", allow_module_level=True)

# Skip all tests in this file if requests is not installed
pytestmark = pytest.mark.skipif(not is_requests_available(), reason="requests package not installed")

MOCK_URL = "http://mock-reward-server.test/api/reward"

@mock.patch('requests.post')
def test_remote_reward_success(mock_post):
    # Configure the mock response for success
    mock_response = mock.Mock()
    mock_response.raise_for_status.return_value = None # Simulate successful HTTP status
    mock_response.json.return_value = {"rewards": [0.95, -0.15]} # Simulate API JSON response
    mock_post.return_value = mock_response

    # Instantiate the client
    client = RemoteRewardClient(reward_url=MOCK_URL)

    # Prepare inputs
    sample_ids = ["remote_id1", "remote_id2"]
    queries = ["query 1", "query 2"]
    responses = ["response 1", "response 2"]

    # Call the method
    rewards = client.get_rewards(sample_ids, queries, responses)

    # Assertions
    mock_post.assert_called_once_with(
        MOCK_URL,
        json={
            "sample_ids": sample_ids,
            "queries": queries,
            "responses": responses
        },
        headers={"Content-Type": "application/json"},
        timeout=30
    )
    assert len(rewards) == 2
    assert torch.isclose(rewards[0], torch.tensor([0.95]))
    assert torch.isclose(rewards[1], torch.tensor([-0.15]))

@mock.patch('requests.post')
def test_remote_reward_http_error(mock_post):
    # Configure the mock to raise an HTTP error
    mock_response = mock.Mock()
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Client Error")
    mock_post.return_value = mock_response

    client = RemoteRewardClient(reward_url=MOCK_URL)
    sample_ids = ["err_id1"]
    queries = ["q_err"]
    responses = ["r_err"]

    # Call the method - should catch the exception and return default rewards
    rewards = client.get_rewards(sample_ids, queries, responses)

    # Assertions
    mock_post.assert_called_once() # Check if called
    assert len(rewards) == 1
    assert torch.isclose(rewards[0], torch.tensor([-1.0])) # Default error reward

@mock.patch('requests.post')
def test_remote_reward_request_exception(mock_post):
    # Configure the mock to raise a general request exception
    mock_post.side_effect = requests.exceptions.RequestException("Connection error")

    client = RemoteRewardClient(reward_url=MOCK_URL)
    sample_ids = ["conn_err_id"]
    queries = ["q_conn_err"]
    responses = ["r_conn_err"]

    # Call the method
    rewards = client.get_rewards(sample_ids, queries, responses)

    # Assertions
    mock_post.assert_called_once()
    assert len(rewards) == 1
    assert torch.isclose(rewards[0], torch.tensor([-1.0])) # Default error reward

@mock.patch('requests.post')
def test_remote_reward_invalid_json(mock_post):
    # Configure the mock response with invalid JSON (missing 'rewards' key)
    mock_response = mock.Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"status": "ok"} # Missing 'rewards'
    mock_post.return_value = mock_response

    client = RemoteRewardClient(reward_url=MOCK_URL)
    sample_ids = ["json_err_id"]
    queries = ["q_json_err"]
    responses = ["r_json_err"]

    # Call the method
    rewards = client.get_rewards(sample_ids, queries, responses)

    # Assertions
    mock_post.assert_called_once()
    assert len(rewards) == 1
    assert torch.isclose(rewards[0], torch.tensor([-1.0])) # Default error reward
