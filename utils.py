from maggma.stores import MongoStore
from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN
import numpy as np


def get_data_wrapper(launchpad_file, query, collections_name='quacc_results0'):
    tasks_store = MongoStore.from_launchpad_file(launchpad_file, collections_name)
    tasks_store.connect()
    docs = list(tasks_store.query(query))
    return docs


def compare_mols(Molecule1, Molecule2):
    molgraph1 = MoleculeGraph.with_local_env_strategy(Molecule1, OpenBabelNN())
    molgraph2 = MoleculeGraph.with_local_env_strategy(Molecule2, OpenBabelNN())
    return molgraph1.isomorphic_to(molgraph2)


def clean_calcs(docs):
    status_array = np.zeros((265, 6))
    # Columns: TS0, IRC_f0, IRC_r0, TS_1, IRC_f1, IRC_r1
    count = 0
    for doc in docs:
        ts_type = int(doc['name'].split('-')[0][-1])
        calc_type = doc['name'][:2]
        index = int(doc['name'].split('-')[-1])
        if calc_type == 'TS':
            status_array[index, 3 * ts_type + 0] = 1
        elif calc_type == 'fo':
            status_array[index, 3 * ts_type + 1] = 1
        elif calc_type == 're':
            status_array[index, 3 * ts_type + 2] = 1
        count = count + 1
    row_indices = np.nonzero(np.all(status_array != 0, axis=1))[0]
    return row_indices


def get_data(indices, launchpad_file, class_tag="sella_ts_prod_jun21_[10]", job_type="TS", ts_type=0, print_level=1):
    all_mols = []
    doc_dict = {}

    query = {
        "metadata.fw_spec.tags.class": {"$regex": class_tag},
        "metadata.tag": {"$in": [job_type + str(ts_type) + "-" + f'{index:03}' for index in indices]}
    }

    docs = get_data_wrapper(launchpad_file, query, collections_name='quacc_results0')
    
    for ii, doc in enumerate(docs):
        geom_index        = doc['metadata']['tag']
        output            = doc['output']
        if job_type == 'TS':
            n_iters           = len(output['ts']['trajectory_results'])
            energy            = output['thermo']['thermo']['results']['energy']
            enthalpy          = output['thermo']['thermo']['results']['enthalpy']
            entropy           = output['thermo']['thermo']['results']['entropy']
            gibbs_free_energy = output['thermo']['thermo']['results']['gibbs_energy']
            zpe               = output['thermo']['thermo']['results']['zpe']
            imag_vib_freqs    = output['thermo']['vib']['results']['imag_vib_freqs']
            molecule_dict     = output['ts']['trajectory'][-1]['molecule']
            mol               = Molecule.from_dict(molecule_dict)
        elif (job_type == 'irc-forward') or (job_type == 'irc-reverse'):
            n_iters           = 0
            energy            = output['irc']['thermo']['thermo']['results']['energy']
            enthalpy          = output['irc']['thermo']['thermo']['results']['enthalpy']
            entropy           = output['irc']['thermo']['thermo']['results']['entropy']
            gibbs_free_energy = output['irc']['thermo']['thermo']['results']['gibbs_energy']
            zpe               = output['irc']['thermo']['thermo']['results']['zpe']
            imag_vib_freqs    = output['irc']['thermo']['vib']['results']['imag_vib_freqs']
            molecule_dict     = output['opt']['trajectory'][-1]['molecule']
            mol               = Molecule.from_dict(molecule_dict)

        # print('ssssssssssssssssssssssssss', geom_index.split('-')[-1], geom_index, job_type, ts_type)

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
        np.savetxt('all_analysis_data' + str(ts_type) + '-' + str(job_type) + '.txt', all_analysis_data, fmt='%.8f')

    return doc_dict, all_analysis_data, all_mols


def get_data1(indices, launchpad_file, class_tag="sella_ts_prod_jun21_[10]", job_type="TS", ts_type=0, print_level=1):
    all_analysis_data = np.zeros((len(indices), 7))
    all_mols = []
    count = 0
    doc_dict = {}
    query = {"metadata.fw_spec.tags.class": {"$regex": class_tag},
             "name": {"$in": [job_type + str(ts_type) + "-" + f'{index:03}' for index in indices]}
    }

    docs = get_data_wrapper(launchpad_file, query, collections_name='quacc_results0')
    for doc in docs:
        geom_index = doc['name'].split('-')[-1]
        n_iters = len(doc['output']['trajectory_results'])
        energy = doc['output']['thermo_results']['energy']
        enthalpy = doc['output']['thermo_results']['enthalpy']
        entropy = doc['output']['thermo_results']['entropy']
        gibbs_free_energy = doc['output']['thermo_results']['gibbs_energy']
        hess_eigvalues = doc['output']['frequencies_real']
        molecule_dict = doc['output']['trajectory'][-1]['molecule']
        mol = Molecule.from_dict(molecule_dict)
        tmp = {}
        tmp["n_iters"] = n_iters
        tmp["energy"] = energy
        tmp["enthalpy"] = enthalpy
        tmp["entropy"] = entropy
        tmp["gibbs_free_energy"] = gibbs_free_energy
        tmp["hess_eigvalues"] = hess_eigvalues
        tmp["molecule_dict"] = molecule_dict
        tmp["mol"] = mol
        doc_dict[int(geom_index)] = tmp
    for index in indices:
        if index in doc_dict:
            all_analysis_data[index, 0] = index
            all_analysis_data[index, 1] = doc_dict[index]["n_iters"]
            all_analysis_data[index, 2] = doc_dict[index]["energy"]
            all_analysis_data[index, 3] = doc_dict[index]["enthalpy"]
            all_analysis_data[index, 4] = doc_dict[index]["entropy"]
            all_analysis_data[index, 5] = doc_dict[index]["gibbs_free_energy"]
            all_analysis_data[index, 6] = np.min(doc_dict[index]["hess_eigvalues"])
        else:
            all_analysis_data[index, 0] = index
        all_mols.append(mol)
    if print_level:
        np.savetxt('all_analysis_data' + str(ts_type) + '-' + str(job_type) + '.txt', all_analysis_data, fmt='%.8f')
    return doc_dict, all_analysis_data, all_mols

