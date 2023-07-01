import os.path

import numpy as np
from maggma.stores import MongoStore
from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN
from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash


def get_data_wrapper(launchpad_file, query, collections_name='quacc_results0'):
    """
    Retrieves documents from a MongoDB collection based on the provided query.

    Args:
        launchpad_file (str): The path to the launchpad file.
        query (dict): The query to filter the documents.
        collections_name (str, optional): The name of the MongoDB collection. Defaults to 'quacc_results0'.

    Returns:
        list: A list of documents that match the query.
    """
    tasks_store = MongoStore.from_launchpad_file(launchpad_file, collections_name)
    tasks_store.connect()
    docs = list(tasks_store.query(query))
    return docs


def compare_mols(molecule_1, molecule_2) -> bool:
    """
    Compare two molecules based on their graph structure.

    Args:
        molecule_1 (pymatgen.core.structure.Molecule): First molecule.
        molecule_2 (pymatgen.core.structure.Molecule): Second molecule.

    Returns:
        bool: True if the molecules have the same graph structure, False otherwise.
    """
    molgraph_1 = create_molecule_graph(molecule_1)
    molgraph_2 = create_molecule_graph(molecule_2)

    graph_1 = molgraph_1.graph.to_undirected()
    graph_2 = molgraph_2.graph.to_undirected()

    add_specie_suffix(graph_1)
    add_specie_suffix(graph_2)

    graph_1_hash = get_graph_hash(graph_1)
    graph_2_hash = get_graph_hash(graph_2)

    return graph_1_hash == graph_2_hash


def create_molecule_graph(molecule):
    """
    Create a molecule graph using the OpenBabelNN strategy.

    Args:
        molecule (pymatgen.core.structure.Molecule): The molecule.

    Returns:
        pymatgen.analysis.graphs.MoleculeGraph: The molecule graph.
    """
    return MoleculeGraph.with_local_env_strategy(molecule, OpenBabelNN())


def add_specie_suffix(graph):
    """
    Add a suffix to each node's 'specie' attribute in the graph.

    Args:
        graph (networkx.Graph): The graph.
    """
    for idx in graph.nodes():
        graph.nodes()[idx]["specie"] = graph.nodes()[idx]["specie"] + str(idx)


def get_graph_hash(graph):
    """
    Get the hash of the graph using the Weisfeiler-Lehman algorithm.

    Args:
        graph (networkx.Graph): The graph.

    Returns:
        str: The graph hash.
    """
    return weisfeiler_lehman_graph_hash(graph, node_attr='specie')


def get_data(indices,
             launchpad_file,
             class_tag="sella_ts_prod_jun21_[10]",
             job_type="TS",
             ts_type=0,
             print_level=1,
             log_dir='logs'):
    all_mols = []
    doc_dict = {}

    query = {
        "metadata.fw_spec.tags.class": {"$regex": class_tag},
        "metadata.tag": {"$in": [job_type + str(ts_type) + "-" + f'{index:03}' for index in indices]}
    }

    docs = get_data_wrapper(launchpad_file, query, collections_name='quacc_results0')

    for doc in docs:
        output = doc['output']
        molecule_dict = {}

        if job_type == 'TS':
            trajectory = output.get('ts', {}).get('trajectory', [])
            if trajectory and isinstance(trajectory, list):
                molecule_dict = trajectory[-1].get('molecule', {})
        elif job_type in ['irc-forward', 'irc-reverse']:
            trajectory = output.get('opt', {}).get('trajectory', [])
            if trajectory and isinstance(trajectory, list):
                molecule_dict = trajectory[-1].get('molecule', {})

        n_iters = len(output.get('ts', {}).get('trajectory_results', []))
        energy = output.get('thermo', {}).get('thermo', {}).get('results', {}).get('energy', 0.0)
        enthalpy = output.get('thermo', {}).get('thermo', {}).get('results', {}).get('enthalpy', 0.0)
        entropy = output.get('thermo', {}).get('thermo', {}).get('results', {}).get('entropy', 0.0)
        gibbs_free_energy = output.get('thermo', {}).get('thermo', {}).get('results', {}).get('gibbs_energy', 0.0)
        zpe = output.get('thermo', {}).get('thermo', {}).get('results', {}).get('zpe', 0.0)
        imag_vib_freqs = output.get('thermo', {}).get('vib', {}).get('results', {}).get('imag_vib_freqs', 0.0)

        doc_dict[int(doc['metadata']['tag'].split('-')[-1])] = {
            "n_iters": n_iters,
            "energy": energy,
            "enthalpy": enthalpy,
            "entropy": entropy,
            "gibbs_free_energy": gibbs_free_energy,
            "zpe": zpe,
            "imag_vib_freq": np.min(imag_vib_freqs) if isinstance(imag_vib_freqs, list) and len(imag_vib_freqs) else 0,
            "molecule_dict": molecule_dict,
            "mol": Molecule.from_dict(molecule_dict)
        }

    all_analysis_data = np.zeros((len(indices), 8))

    for index in indices:
        if index in doc_dict:
            data = doc_dict[index]
            all_analysis_data[index, :] = [
                index,
                data["n_iters"],
                data["energy"],
                data["enthalpy"],
                data["entropy"],
                data["gibbs_free_energy"],
                data["zpe"],
                data["imag_vib_freq"]
            ]
            all_mols.append(data["mol"])
        else:
            all_analysis_data[index, 0] = index

    if print_level:
        os.makedirs(log_dir, exist_ok=True)
        np.savetxt(os.path.join(log_dir, f"all_analysis_data{ts_type}-{job_type}.txt"),
                   all_analysis_data,
                   fmt='%.8f')

    return doc_dict, all_analysis_data, all_mols
