import pytest
from main import retrieve_data


@pytest.fixture
def mock_get_data(mocker):
    def mock_return(indices, lp_file, class_tag, ts_type, job_type):
        # Create a mock dictionary for the returned data
        data = {
            "mol": {"mock_data": True},
            "gibbs_free_energy": 10.0,
            "n_iters": 5,
            "imag_vib_freq": [100, 200, 300]
        }
        return {index: data for index in indices}

    # Patch the get_data function with the mock_return function
    mocker.patch("main.get_data", side_effect=mock_return)

def test_retrieve_data(mock_get_data):
    lp_file = "test_file.yaml"
    tag = "test_tag"
    indices = [0, 1, 2]

    # Call the retrieve_data function
    result = retrieve_data(lp_file, tag, indices)

    # Check the expected structure of the result
    assert isinstance(result, dict)
    assert set(result.keys()) == {0, 1}
    assert set(result[0].keys()) == {"TS", "firc", "rirc"}
    assert set(result[1].keys()) == {"TS", "firc", "rirc"}

    # Check the content of the returned data for a specific index
    assert isinstance(result[0]["TS"][0], dict)
    assert result[0]["TS"][0]["mol"]["mock_data"] == True
    assert result[0]["TS"][0]["gibbs_free_energy"] == 10.0
    assert result[0]["TS"][0]["n_iters"] == 5
    assert result[0]["TS"][0]["imag_vib_freq"] == [100, 200, 300]
