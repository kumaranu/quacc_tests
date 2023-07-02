import os
import numpy as np
import pytest
from pymatgen.core.structure import Molecule
import networkx as nx
from utils import (
    compare_mols,
    create_molecule_graph,
    add_specie_suffix,
    get_graph_hash,
    get_data,
    get_data_wrapper
)
from main import (
    retrieve_data,
    check_present_indices,
    compare_mols,
    perform_comparisons,
)


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


@pytest.fixture
def sample_molecule():
    molecule_dict = {"@module": "pymatgen.core.structure",
                     "@class": "Molecule",
                     "charge": 0,
                     "spin_multiplicity": 1,
                     "sites": [
                         {"name": "C",
                          "species": [{"element": "C", "occu": 1}],
                          "xyz": [1.4165827672103317, -0.3373093639887552, 0.9216694245355858],
                          "properties": {"magmom": 0.0,
                                         "masses": 12.0}},
                         {"name": "C",
                          "species": [{"element": "C", "occu": 1}],
                          "xyz": [0.32745586199762866, 0.5716506996310059, 1.22573442024535],
                          "properties": {"magmom": 0.0, "masses": 12.0}},
                         {"name": "C",
                          "species": [{"element": "C", "occu": 1}],
                          "xyz": [-1.160644243188324, 0.23271399012112526, 1.03816854724771],
                          "properties": {"magmom": 0.0, "masses": 12.0}},
                         {"name": "C",
                          "species": [{"element": "C", "occu": 1}],
                          "xyz": [-1.5878095357987172, -0.9275715486308965, 1.9430570310617898],
                          "properties": {"magmom": 0.0, "masses": 12.0}},
                         {"name": "C",
                          "species": [{"element": "C", "occu": 1}],
                          "xyz": [-1.2755566560089167, -0.6966991135644678, 3.420326287934827],
                          "properties": {"magmom": 0.0, "masses": 12.0}},
                         {"name": "H",
                          "species": [{"element": "H", "occu": 1}],
                          "xyz": [0.7258422602475881, 1.1936706921605928, 0.3689816066916836],
                          "properties": {"magmom": 0.0, "masses": 1.00782503223}},
                         {"name": "H", "species": [{"element": "H", "occu": 1}],
                          "xyz": [0.9955190014988713, -1.0729948235831677, 0.19725084901083573],
                          "properties": {"magmom": 0.0, "masses": 1.00782503223}},
                         {"name": "H",
                          "species": [{"element": "H", "occu": 1}],
                          "xyz": [0.5057682556745512, 1.2017212717830226, 2.107533464579369],
                          "properties": {"magmom": 0.0, "masses": 1.00782503223}},
                         {"name": "H",
                          "species": [{"element": "H", "occu": 1}],
                          "xyz": [-1.767317673105808, 1.117824634125562, 1.272266203632612],
                          "properties": {"magmom": 0.0, "masses": 1.00782503223}},
                         {"name": "H",
                          "species": [{"element": "H", "occu": 1}],
                          "xyz": [-1.3608250123541785, -0.029173273614768933, -0.00889162908158537],
                          "properties": {"magmom": 0.0, "masses": 1.00782503223}},
                         {"name": "H",
                          "species": [{"element": "H", "occu": 1}],
                          "xyz": [-2.663741338424574, -1.098546188159488, 1.8134282599890403],
                          "properties": {"magmom": 0.0, "masses": 1.00782503223}},
                         {"name": "H",
                          "species": [{"element": "H", "occu": 1}],
                          "xyz": [-1.0884591520537563, -1.8472338718768546, 1.6093170999009994],
                          "properties": {"magmom": 0.0, "masses": 1.00782503223}},
                         {"name": "H",
                          "species": [{"element": "H", "occu": 1}],
                          "xyz": [-1.7106263507546275, 0.24518370567293787, 3.776686976865684],
                          "properties": {"magmom": 0.0, "masses": 1.00782503223}},
                         {"name": "H",
                          "species": [{"element": "H", "occu": 1}],
                          "xyz": [-1.6759010963544012, -1.5062440985918497, 4.03937789625853],
                          "properties": {"magmom": 0.0, "masses": 1.00782503223}},
                         {"name": "H",
                          "species": [{"element": "H", "occu": 1}],
                          "xyz": [-0.19411813694794594, -0.6482981133373051, 3.593708646307539],
                          "properties": {"magmom": 0.0, "masses": 1.00782503223}}]}
    return Molecule.from_dict(molecule_dict)


def test_compare_mols_same_molecule(sample_molecule):
    # Test case where the same molecule is compared
    assert compare_mols(sample_molecule, sample_molecule) is True


def test_compare_mols_different_molecules(sample_molecule):
    different_molecule_dict = {
        "@module": "pymatgen.core.structure",
        "@class": "Molecule",
        "charge": 0,
        "spin_multiplicity": 1,
        "sites": [
            {"name": "C",
             "species": [
                 {"element": "C",
                  "occu": 1}
             ],
             "xyz": [1.4008478771935395, -0.3538513449641797, 0.9503761111607852],
             "properties": {"magmom": 0.0,
                            "masses": 12.0}
             },
            {"name": "C",
             "species": [
                 {"element": "C",
                  "occu": 1}
             ],
             "xyz": [0.32199078453847424, 0.576363397461615, 1.224498187534653],
             "properties": {"magmom": 0.0,
                            "masses": 12.0}
             },
            {"name": "C",
             "species": [
                 {"element": "C",
                  "occu": 1}
             ],
             "xyz": [-1.1682939091718492, 0.24739254422249657, 1.0355079929148916],
             "properties": {"magmom": 0.0,
                            "masses": 12.0}
             },
            {"name": "C",
             "species": [
                 {"element": "C",
                  "occu": 1}
             ],
             "xyz": [-1.5844327066167663, -0.9197430596763388, 1.9369646511487582],
             "properties": {"magmom": 0.0,
                            "masses": 12.0}
             },
            {"name": "C",
             "species": [
                 {"element": "C",
                  "occu": 1}
             ],
             "xyz": [-1.267938300577513, -0.6990227710873599, 3.4136440080507766],
             "properties": {"magmom": 0.0,
                            "masses": 12.0}
             },
            {"name": "H",
             "species": [
                 {"element": "H",
                  "occu": 1}
             ],
             "xyz": [0.7384760705476303, 1.1757993015880042, 0.36089528763784656],
             "properties": {"magmom": 0.0,
                            "masses": 1.00782503223}
             },
            {"name": "H",
             "species": [
                 {"element": "H",
                  "occu": 1}
             ],
             "xyz": [0.9753415577682765, -1.1013144392685414, 0.24062050657933817],
             "properties": {"magmom": 0.0,
                            "masses": 1.00782503223}
             },
            {"name": "H",
             "species": [
                 {"element": "H",
                  "occu": 1}
             ],
             "xyz": [0.5004394002367588, 1.2189531371264337, 2.0971542342831833],
             "properties": {"magmom": 0.0,
                            "masses": 1.00782503223}
             },
            {"name": "H",
             "species": [
                 {"element": "H",
                  "occu": 1}
             ],
             "xyz": [-1.7733738606235712, 1.131676507741109, 1.2736441547551254],
             "properties": {"magmom": 0.0,
                            "masses": 1.00782503223}
             },
            {"name": "H",
             "species": [
                 {"element": "H",
                  "occu": 1}
             ],
             "xyz": [-1.3727435892674373, -0.014787771025894758, -0.010736089150883272],
             "properties": {"magmom": 0.0,
                            "masses": 1.00782503223}
             },
            {"name": "H",
             "species": [
                 {"element": "H",
                  "occu": 1}
             ],
             "xyz": [-2.6584897496506974, -1.10452433010817, 1.807971737509261],
             "properties": {"magmom": 0.0,
                            "masses": 1.00782503223}
             },
            {"name": "H",
             "species": [
                 {"element": "H",
                  "occu": 1}
             ],
             "xyz": [-1.0729158458762569, -1.83035989953293, 1.596564638848291],
             "properties": {"magmom": 0.0,
                            "masses": 1.00782503223}
             },
            {"name": "H",
             "species": [{"element": "H",
                          "occu": 1}
                         ],
             "xyz": [-1.7195893134418407, 0.22864921166459407, 3.786067660787342],
             "properties": {"magmom": 0.0,
                            "masses": 1.00782503223}
             },
            {"name": "H",
             "species": [{"element": "H",
                          "occu": 1}
                         ],
             "xyz": [-1.6466710914126845, -1.5252109960394395, 4.0246253756221515],
             "properties": {"magmom": 0.0,
                            "masses": 1.00782503223}
             },
            {"name": "H",
             "species": [{"element": "H",
                          "occu": 1}
                         ],
             "xyz": [-0.18647837200834205, -0.6313248899547064, 3.5808166274984505],
             "properties": {"magmom": 0.0,
                            "masses" : 1.00782503223}
             }
        ]
    }
    different_molecule = Molecule.from_dict(different_molecule_dict)
    assert compare_mols(sample_molecule, different_molecule) is True


def test_create_molecule_graph(sample_molecule):
    # Test creation of molecule graph
    molgraph = create_molecule_graph(sample_molecule)
    assert molgraph is not None


def test_add_specie_suffix():
    # Create a sample graph
    graph = nx.Graph()
    graph.add_node(0, specie="H")
    graph.add_node(1, specie="O")

    # Add suffixes
    add_specie_suffix(graph)

    # Check if suffixes are added correctly
    assert graph.nodes[0]["specie"] == "H0"
    assert graph.nodes[1]["specie"] == "O1"


def test_get_graph_hash():
    # Create a sample graph
    graph = nx.Graph()
    graph.add_node(0, specie="H")
    graph.add_node(1, specie="O")
    graph.add_edge(0, 1)

    # Get the hash
    graph_hash = get_graph_hash(graph)

    # Check if hash is obtained
    assert graph_hash is not None


def test_get_data():
    # Set up the test data
    indices = [1, 2, 4]
    launchpad_file = os.path.join(os.environ["HOME"], "fw_config/my_launchpad.yaml")
    class_tag = "sella_ts_prod_jun25_[10]"
    job_type = "TS"
    ts_type = 0
    print_level = 1
    log_dir = 'logs'

    # Call the function to get the results
    doc_dict, all_analysis_data, all_mols = get_data(
        indices,
        launchpad_file,
        class_tag=class_tag,
        job_type=job_type,
        ts_type=ts_type,
        print_level=print_level,
        log_dir=log_dir
    )

    print('doc_dict.keys():', doc_dict.keys())
    # Perform assertions to check the correctness of the results
    assert len(doc_dict) == len(indices)  # Check the length of the doc_dict
    #assert all_analysis_data.shape == (len(indices), 8)  # Check the shape of all_analysis_data
    #assert len(all_mols) == len(indices)  # Check the length of all_mols


def test_retrieve_data():
    lp_file = os.path.join(os.environ["HOME"], "fw_config/my_launchpad.yaml")
    tag = "sella_ts_prod_jun25_[10]"
    indices = np.arange(5)
    master_dict = retrieve_data(lp_file, tag, indices)

    assert isinstance(master_dict, dict)
    assert len(master_dict) == 2

    for ts_type in master_dict:
        assert isinstance(master_dict[ts_type], dict)
        assert len(master_dict[ts_type]) == 3

        for calc_type in master_dict[ts_type]:
            assert isinstance(master_dict[ts_type][calc_type], dict)
