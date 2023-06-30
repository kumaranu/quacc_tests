import os
import pytest
from utils import get_data_wrapper


@pytest.mark.test
def test_get_data_wrapper():
    launchpad_file = os.path.join(os.environ["HOME"], "fw_config/my_launchpad.yaml")
    class_tag = "sella_ts_prod_jun25_[10]"
    query = {
        "metadata.fw_spec.tags.class": {"$regex": class_tag},
        "metadata.tag": 'TS0-000'
    }
    docs = get_data_wrapper(launchpad_file, query, collections_name='quacc_results0')
    # Assertions
    assert 'ts' in docs[0]['output'].keys()

