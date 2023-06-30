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


def compare_mols(molecule1, molecule2):
    molgraph1 = MoleculeGraph.with_local_env_strategy(molecule1, OpenBabelNN())
    molgraph2 = MoleculeGraph.with_local_env_strategy(molecule2, OpenBabelNN())
    graph1 = molgraph1.graph.to_undirected()
    graph2 = molgraph2.graph.to_undirected()
    for idx in graph1.nodes():
        graph1.nodes()[idx]["specie"] = graph1.nodes()[idx]["specie"] + str(idx)
    for idx in graph2.nodes():
        graph2.nodes()[idx]["specie"] = graph2.nodes()[idx]["specie"] + str(idx)
    graph1_hash = weisfeiler_lehman_graph_hash(graph1, node_attr='specie')
    graph2_hash = weisfeiler_lehman_graph_hash(graph2, node_attr='specie')
    return graph1_hash == graph2_hash


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
        mol = None

        if job_type == 'TS':
            trajectory = output.get('ts', {}).get('trajectory', [])
            if trajectory and isinstance(trajectory, list):
                molecule_dict = trajectory[-1].get('molecule', {})
        elif job_type in ['irc-forward', 'irc-reverse']:
            trajectory = output.get('opt', {}).get('trajectory', [])
            if trajectory and isinstance(trajectory, list):
                molecule_dict = trajectory[-1].get('molecule', {})

        mol = Molecule.from_dict(molecule_dict)

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
            "mol": mol
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
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        np.savetxt(log_dir + '/all_analysis_data' + str(ts_type) + '-' + str(job_type) + '.txt',
                   all_analysis_data,
                   fmt='%.8f')

    return doc_dict, all_analysis_data, all_mols


def get_data1(indices, launchpad_file, class_tag="sella_ts_prod_jun21_[10]", job_type="TS", ts_type=0, print_level=1):
    all_mols = []
    doc_dict = {}

    query = {
        "metadata.fw_spec.tags.class": {"$regex": class_tag},
        "metadata.tag": {"$in": [job_type + str(ts_type) + "-" + f'{index:03}' for index in indices]}
    }

    docs = get_data_wrapper(launchpad_file, query, collections_name='quacc_results0')

    for ii, doc in enumerate(docs):
        n_iters = 0
        energy = 0.0
        enthalpy = 0.0
        entropy = 0.0
        gibbs_free_energy = 0.0
        zpe = 0.0
        imag_vib_freqs = 0.0
        molecule_dict = {}
        mol = None
        geom_index = doc['metadata']['tag']
        output = doc['output']
        if job_type == 'TS':
            n_iters = len(output['ts']['trajectory_results'])
            energy = output['thermo']['thermo']['results']['energy']
            enthalpy = output['thermo']['thermo']['results']['enthalpy']
            entropy = output['thermo']['thermo']['results']['entropy']
            gibbs_free_energy = output['thermo']['thermo']['results']['gibbs_energy']
            zpe = output['thermo']['thermo']['results']['zpe']
            imag_vib_freqs = output['thermo']['vib']['results']['imag_vib_freqs']
            molecule_dict = output['ts']['trajectory'][-1]['molecule']
            mol = Molecule.from_dict(molecule_dict)
        elif (job_type == 'irc-forward') or (job_type == 'irc-reverse'):
            n_iters = 0
            energy = output['irc']['thermo']['thermo']['results']['energy']
            enthalpy = output['irc']['thermo']['thermo']['results']['enthalpy']
            entropy = output['irc']['thermo']['thermo']['results']['entropy']
            gibbs_free_energy = output['irc']['thermo']['thermo']['results']['gibbs_energy']
            zpe = output['irc']['thermo']['thermo']['results']['zpe']
            imag_vib_freqs = output['irc']['thermo']['vib']['results']['imag_vib_freqs']
            molecule_dict = output['opt']['trajectory'][-1]['molecule']
            mol = Molecule.from_dict(molecule_dict)

        doc_dict[int(geom_index.split('-')[-1])] = {
            "n_iters": n_iters,
            "energy": energy,
            "enthalpy": enthalpy,
            "entropy": entropy,
            "gibbs_free_energy": gibbs_free_energy,
            "zpe": zpe,
            "imag_vib_freq": np.min(imag_vib_freqs) if len(imag_vib_freqs) else 0,
            "molecule_dict": molecule_dict,
            "mol": mol
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
        np.savetxt('all_analysis_data' + str(ts_type) + '-' + str(job_type) + '.txt',
                   all_analysis_data,
                   fmt='%.8f')

    return doc_dict, all_analysis_data, all_mols
