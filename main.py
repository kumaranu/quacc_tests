import json
import os
import ase.io
from ase import Atoms
import numpy as np
from numpy import ndarray
import seaborn as sns
import matplotlib.pyplot as plt

from utils import compare_mols, get_data
from typing import List, Dict, Set
from visuals import process_trajectories
from utils import get_data_wrapper


def retrieve_data(lp_file, tag, indices):
    dd_ts = {}
    dd_firc = {}
    dd_rirc = {}
    for ts_type in [0, 1]:
        dd_ts[ts_type], _, _ = get_data(indices, lp_file, class_tag=tag, ts_type=ts_type, job_type='TS')
        dd_firc[ts_type], _, _ = get_data(indices, lp_file, class_tag=tag, ts_type=ts_type, job_type='irc-forward')
        dd_rirc[ts_type], _, _ = get_data(indices, lp_file, class_tag=tag, ts_type=ts_type, job_type='irc-reverse')

    master_dict = {0: {"TS": dd_ts[0], "firc": dd_firc[0], "rirc": dd_rirc[0]},
                   1: {"TS": dd_ts[1], "firc": dd_firc[1], "rirc": dd_rirc[1]}}

    return master_dict


def check_present_indices(master_dict, indices):
    set_err_0, set_err_1, set_err_both = set(), set(), set()

    good_indices = []
    for index in indices:
        all_present = {0: True, 1: True}
        for ts_type in [0, 1]:
            for calc_type in ["TS", "firc", "rirc"]:
                if index not in master_dict[ts_type][calc_type]:
                    if ts_type == 0:
                        set_err_0.add((index, calc_type))
                    elif ts_type == 1:
                        set_err_1.add((index, calc_type))
                    all_present[ts_type] = False
        if all(all_present.values()):
            good_indices.append(index)
    failed_index_count0 = len(set([i[0] for i in set_err_0]))
    failed_index_count1 = len(set([i[0] for i in set_err_1]))
    failed_index_count10 = len(set([i[0] for i in set_err_0.intersection(set_err_1)]))

    print(f'\nFailed calculation counts for type-0 and type-1: ',
          f'{failed_index_count0}, {failed_index_count1}')
    print(f'Failed calculations for type 0: {failed_index_count0}: {set_err_0}')
    print(f'Failed calculations for type 1: {failed_index_count1}: {set_err_1}')
    print(f'Failed calculations for both: {failed_index_count10}: {set_err_1.intersection(set_err_0)}')

    print(f'\nNumber of all-good indices: {len(good_indices)}')
    return good_indices


def perform_comparisons(
        master_dict: Dict[int, Dict[str, List[Dict[str, any]]]],
        good_indices: List[int],
        imag_freq_threshold: float,
        delta_g_threshold: float
) -> tuple[
    set[int], set[int], dict[int, int], dict[int, int], set[int], set[int], set[int], set[int], set[int], ndarray[
        float]]:
    iter_comparison1: Dict[int, int] = {0: 0, 1: 0}
    iter_comparison2: Dict[int, int] = {0: 0, 1: 0}
    set_same_rxn: Set[int] = set()
    set_diff_rxn: Set[int] = set()
    set_imag_freqs: Set[int] = set()
    set_delta_g_f: Set[int] = set()
    set_delta_g_r: Set[int] = set()
    set_no_rxn0: Set[int] = set()
    set_no_rxn1: Set[int] = set()
    general_data: np.ndarray[float] = np.zeros((len(good_indices), 17))

    for ii, index in enumerate(good_indices):
        check0f0r = compare_mols(master_dict[0]["firc"][index]["mol"], master_dict[0]["rirc"][index]["mol"])
        check1f1r = compare_mols(master_dict[1]["firc"][index]["mol"], master_dict[1]["rirc"][index]["mol"])
        check0f1f = compare_mols(master_dict[0]["firc"][index]["mol"], master_dict[1]["firc"][index]["mol"])
        check0r1r = compare_mols(master_dict[0]["rirc"][index]["mol"], master_dict[1]["rirc"][index]["mol"])
        check0f1r = compare_mols(master_dict[0]["firc"][index]["mol"], master_dict[1]["rirc"][index]["mol"])
        check1f0r = compare_mols(master_dict[1]["firc"][index]["mol"], master_dict[0]["rirc"][index]["mol"])

        gibbs_ts0 = master_dict[0]["TS"][index]["gibbs_free_energy"]
        gibbs_f0 = master_dict[0]["firc"][index]["gibbs_free_energy"]
        gibbs_r0 = master_dict[0]["rirc"][index]["gibbs_free_energy"]

        gibbs_ts1 = master_dict[1]["TS"][index]["gibbs_free_energy"]
        gibbs_f1 = master_dict[1]["firc"][index]["gibbs_free_energy"]
        gibbs_r1 = master_dict[1]["rirc"][index]["gibbs_free_energy"]

        imag_freq0 = np.min(master_dict[0]["TS"][index]["imag_vib_freq"])
        imag_freq1 = np.min(master_dict[1]["TS"][index]["imag_vib_freq"])

        e_std_min = np.min(master_dict[0]['TS'][index]['energy_std_ts_traj_list'])
        e_std_max = np.max(master_dict[0]['TS'][index]['energy_std_ts_traj_list'])
        e_std_avg = np.mean(master_dict[0]['TS'][index]['energy_std_ts_traj_list'])
        e_std_last = master_dict[0]['TS'][index]['energy_std_ts_traj_list'][-1]

        # Reactant and product have same bonding for type 0
        if check0f0r:
            set_no_rxn0.add(index)

            # Forward and reverse energy barriers for type 0
            delta_g0_f = gibbs_ts0 - gibbs_f0
            delta_g0_r = gibbs_ts0 - gibbs_r0

            delta_g1_f = gibbs_ts1 - gibbs_f1
            delta_g1_r = gibbs_ts1 - gibbs_r1
            general_data[ii, :] = [index,
                                   check0f0r,
                                   check1f1r,
                                   (check0f1f and check0r1r) or (check0f1r and check1f0r),
                                   imag_freq0, imag_freq1, abs(imag_freq0 - imag_freq1),
                                   delta_g0_f, delta_g1_f, abs(delta_g0_f - delta_g1_f),
                                   delta_g0_r, delta_g1_r, abs(delta_g0_r - delta_g1_r),
                                   e_std_min, e_std_max, e_std_avg, e_std_last]
        # Reactant and product have same bonding for type 1
        if check1f1r:
            set_no_rxn1.add(index)

            # Forward and reverse energy barriers for type 0
            delta_g0_f = gibbs_ts0 - gibbs_f0
            delta_g0_r = gibbs_ts0 - gibbs_r0

            delta_g1_f = gibbs_ts1 - gibbs_f1
            delta_g1_r = gibbs_ts1 - gibbs_r1

            general_data[ii, :] = [index,
                                   check0f0r,
                                   check1f1r,
                                   (check0f1f and check0r1r) or (check0f1r and check1f0r),
                                   imag_freq0, imag_freq1, abs(imag_freq0 - imag_freq1),
                                   delta_g0_f, delta_g1_f, abs(delta_g0_f - delta_g1_f),
                                   delta_g0_r, delta_g1_r, abs(delta_g0_r - delta_g1_r),
                                   e_std_min, e_std_max, e_std_avg, e_std_last]

        # Reactant and product have different bonding in both type 0 and type 1
        if not check0f0r and not check1f1r:
            # --------------------------------------------- #
            # ------- Everything here is a reaction ------- #
            # --- connecting different molecular systems -- #
            # --------------------------------------------- #

            # The reactant from type 0 and the reactant from type 1 have the same graph
            if check0f1f and check0r1r:
                set_same_rxn.add(index)

                iter_comparison1[0] += master_dict[0]["TS"][index]["n_iters1"]
                iter_comparison1[1] += master_dict[1]["TS"][index]["n_iters1"]
                iter_comparison2[0] += master_dict[0]["TS"][index]["n_iters2"]
                iter_comparison2[1] += master_dict[1]["TS"][index]["n_iters2"]

                # Forward and reverse energy barriers for type 0
                delta_g0_f = gibbs_ts0 - gibbs_f0
                delta_g0_r = gibbs_ts0 - gibbs_r0

                # Forward and reverse energy barriers for type 1
                delta_g1_f = gibbs_ts1 - gibbs_f1
                delta_g1_r = gibbs_ts1 - gibbs_r1

                general_data[ii, :] = [index,
                                       check0f0r,
                                       check1f1r,
                                       (check0f1f and check0r1r) or (check0f1r and check1f0r),
                                       imag_freq0, imag_freq1, abs(imag_freq0 - imag_freq1),
                                       delta_g0_f, delta_g1_f, abs(delta_g0_f - delta_g1_f),
                                       delta_g0_r, delta_g1_r, abs(delta_g0_r - delta_g1_r),
                                       e_std_min, e_std_max, e_std_avg, e_std_last]

                # Imaginary frequency is differing more than a threshold
                if abs(imag_freq0 - imag_freq1) > imag_freq_threshold:
                    set_imag_freqs.add(index)
                # Delta G forward is differing more than a threshold
                if abs(delta_g0_f - delta_g1_f) > delta_g_threshold:
                    set_delta_g_f.add(index)
                # Delta G reverse is differing more than a threshold
                if abs(delta_g0_r - delta_g1_r) > delta_g_threshold:
                    set_delta_g_r.add(index)
            # Reactants and products are switched between type 0 and type 1
            elif check0f1r and check1f0r:
                set_same_rxn.add(index)

                iter_comparison1[0] += master_dict[0]["TS"][index]["n_iters1"]
                iter_comparison1[1] += master_dict[1]["TS"][index]["n_iters1"]
                iter_comparison2[0] += master_dict[0]["TS"][index]["n_iters2"]
                iter_comparison2[1] += master_dict[1]["TS"][index]["n_iters2"]

                # Forward and reverse energy barriers for type 0
                delta_g0_f = gibbs_ts0 - gibbs_f0
                delta_g0_r = gibbs_ts0 - gibbs_r0

                # ------------------------------------------------ #
                # ------- Using reverse IRC for Forward ---------- #
                # --------- and forward IRC for reverse ---------- #
                # ---- to calculate energy barriers for type 1 --- #
                # ------------------------------------------------ #
                delta_g1_f = gibbs_ts1 - gibbs_r1
                delta_g1_r = gibbs_ts1 - gibbs_f1

                general_data[ii, :] = [index,
                                       check0f0r,
                                       check1f1r,
                                       (check0f1f and check0r1r) or (check0f1r and check1f0r),
                                       imag_freq0, imag_freq1, abs(imag_freq0 - imag_freq1),
                                       delta_g0_f, delta_g1_f, abs(delta_g0_f - delta_g1_f),
                                       delta_g0_r, delta_g1_r, abs(delta_g0_r - delta_g1_r),
                                       e_std_min, e_std_max, e_std_avg, e_std_last]

                # Imaginary frequency is differing more than a threshold
                if abs(imag_freq0 - imag_freq1) > imag_freq_threshold:
                    set_imag_freqs.add(index)
                # Delta G forward is differing more than a threshold
                if abs(delta_g0_f - delta_g1_f) > delta_g_threshold:
                    set_delta_g_f.add(index)
                # Delta G reverse is differing more than a threshold
                if abs(delta_g0_r - delta_g1_r) > delta_g_threshold:
                    set_delta_g_r.add(index)
            else:
                set_diff_rxn.add(index)

                # Forward and reverse energy barriers for type 0
                delta_g0_f = gibbs_ts0 - gibbs_f0
                delta_g0_r = gibbs_ts0 - gibbs_r0
                if check0f1r or check1f0r:
                    # ------------------------------------------------#
                    # ------- Using reverse IRC for Forward ----------#
                    # --------- and forward IRC for reverse ----------#
                    # ---- to calculate energy barriers for type 1 ---#
                    # ------------------------------------------------#
                    delta_g1_f = gibbs_ts1 - gibbs_r1
                    delta_g1_r = gibbs_ts1 - gibbs_f1
                else:
                    # Forward and reverse energy barriers for type 1
                    delta_g1_f = gibbs_ts1 - gibbs_f1
                    delta_g1_r = gibbs_ts1 - gibbs_r1

                general_data[ii, :] = [index,
                                       check0f0r,
                                       check1f1r,
                                       (check0f1f and check0r1r) or (check0f1r and check1f0r),
                                       imag_freq0, imag_freq1, abs(imag_freq0 - imag_freq1),
                                       delta_g0_f, delta_g1_f, abs(delta_g0_f - delta_g1_f),
                                       delta_g0_r, delta_g1_r, abs(delta_g0_r - delta_g1_r),
                                       e_std_min, e_std_max, e_std_avg, e_std_last]
    return set_no_rxn0, set_no_rxn1, iter_comparison1, iter_comparison2, set_same_rxn, set_diff_rxn,\
        set_imag_freqs, set_delta_g_f, set_delta_g_r, general_data


def traj_arr_to_atoms_list(traj_array):
    list_atoms = []
    for ii, conf in enumerate(traj_array):
        data = json.loads(conf['atoms']['atoms_json'])
        numbers = np.array(data["numbers"]["__ndarray__"][2])
        positions = np.array(data["positions"]["__ndarray__"][2])
        atoms = Atoms(numbers=numbers,
                      positions=np.reshape(positions, (len(numbers), 3)))
        list_atoms.append(atoms)
    return list_atoms


def log_transition_states(indices, master_dict):
        dir_name = "all_transition_states"
        os.makedirs(dir_name, exist_ok=True)
        os.chdir(dir_name)
        for ts_type in [0, 1]:
            os.makedirs(str(ts_type), exist_ok=True)
            os.chdir(str(ts_type))
            for index in indices:
                try:
                    conf = master_dict[ts_type]['TS'][index]['trajectory'][-1]
                    data = json.loads(conf['atoms']['atoms_json'])
                    numbers = np.array(data["numbers"]["__ndarray__"][2])
                    positions = np.array(data["positions"]["__ndarray__"][2])
                    atoms = Atoms(
                        numbers=numbers,
                        positions=np.reshape(positions, (len(numbers), 3))
                    )
                    ase.io.write(f'ts{ts_type}_{index:03}.xyz', atoms)
                except Exception as e:
                    print("TS: an error occurred while accessing transition state for index:", e)
            os.chdir('../')
        os.chdir('../')


def log_trajectories(indices, master_dict):
    dir_name = "all_trajectories"
    os.makedirs(dir_name, exist_ok=True)
    os.chdir(dir_name)
    for ts_type in [0, 1]:
        os.makedirs(str(ts_type), exist_ok=True)
        os.chdir(str(ts_type))
        for calc_type in ['TS', 'firc', 'rirc']:
            os.makedirs(calc_type, exist_ok=True)
            os.chdir(calc_type)
            for index in indices:
                os.makedirs(f'{index:03}', exist_ok=True)
                os.chdir(f'{index:03}')
                if calc_type == 'TS':
                    try:
                        traj_array = master_dict[ts_type][calc_type][index]['trajectory']
                        ase.io.write(calc_type + '.xyz', traj_arr_to_atoms_list(traj_array))
                        process_trajectories(os.getcwd())
                    except Exception as e:
                        print("TS: an error occurred while accessing trajectory for index:", e)
                elif calc_type == 'firc' or calc_type == 'rirc':
                    try:
                        traj_array = master_dict[ts_type][calc_type][index]['trajectory']
                        ase.io.write('opt_qirc.xyz', traj_arr_to_atoms_list(traj_array))
                        process_trajectories(os.getcwd())
                    except Exception as e:
                        print("IRC: an error occurred while accessing trajectory for index:", e)
                os.chdir('../')
            os.chdir('../')
        os.chdir('../')
    os.chdir('../')


def plot_correlations(general_data,
                      columns_to_compare=None):
    # Calculate correlation matrix
    if columns_to_compare is None:
        columns_to_compare = [1, 2, 3, 6, 9, 12, 16]
    correlation_matrix = np.corrcoef(general_data[:, columns_to_compare],
                                     rowvar=False)
    # print('correlation_matrix:\n', correlation_matrix)

    row_labels = [
        'Isomorphism 0',
        'Isomorphism 1',
        'Same reaction/TS',
        'Imag-freq diff.',
        'Delta G forward',
        'Delta G reverse',
        'Energy std. dev.'
    ]
    sns.heatmap(correlation_matrix,
                annot=True,
                cmap='coolwarm')
    plt.xticks(
        np.arange(correlation_matrix.shape[1]) + 0.5,
        row_labels,
        rotation=90
    )
    plt.yticks(
        np.arange(correlation_matrix.shape[0]) + 0.5,
        row_labels,
        rotation=0
    )
    plt.title('Correlation Matrix Heatmap')
    plt.rcParams['figure.figsize'] = [8, 6]
    plt.tight_layout()
    plt.show()


def main():
    lp_file = os.path.join(os.environ["HOME"], "fw_config/my_launchpad.yaml")
    # tag = "sella_ts_prod_jun25_[10]"
    # tag = "sella_ts_prod_jul2b_[10]"
    tag = "sella_ts_prod_jul13d_[10]"

    # Modify the indices based on your requirements
    indices = np.arange(265)

    # Modify the threshold values based on your requirements
    imag_freq_threshold = 10
    delta_g_threshold = 0.0285

    master_dict = retrieve_data(lp_file, tag, indices)

    # log_trajectories(indices, master_dict)
    # log_transition_states(indices, master_dict)

    good_indices = check_present_indices(master_dict, indices)

    set_no_rxn0, set_no_rxn1, iter_comparison1, iter_comparison2, set_same_rxn, set_diff_rxn, set_imag_freqs,\
        set_delta_g_f, set_delta_g_r, general_data = perform_comparisons(master_dict,
                                                                         good_indices,
                                                                         imag_freq_threshold,
                                                                         delta_g_threshold)

    # plot_correlations(general_data)

    print(f"\nset no reaction 0: {len(set_no_rxn0)}: {set_no_rxn0}")
    # for item in set_failed0:
    #     print(f"({item[0]:>3d}, {item[1]:>8.2f}, {item[2]:>6.2f}, {item[3]:>6.2f})")
    print(f"set no reaction 1: {len(set_no_rxn1)}: {set_no_rxn1}")
    # for item in set_failed1:
    #     print(f"({item[0]:>3d}, {item[1]:>8.2f}, {item[2]:>6.2f}, {item[3]:>6.2f})")
    print(f"set both 1 and 0 did not have a reaction: {len(set_no_rxn0.intersection(set_no_rxn1))}:"
          f" {set_no_rxn0.intersection(set_no_rxn1)}")

    print(f"\nset_same_rxn: {len(set_same_rxn)}: {set_same_rxn}")
    print(f"set_diff_rxn: {len(set_diff_rxn)}: {set_diff_rxn}")

    print(f"\nDifferent Imaginary Frequency Numbers: {len(set_imag_freqs)}: {set_imag_freqs}")
    print(f"Different DeltaG (forward) Numbers: {len(set_delta_g_f)}: {set_delta_g_f}")
    print(f"Different DeltaG (reverse) Numbers: {len(set_delta_g_r)}: {set_delta_g_r}")

    print(f"\nIteration Comparison1: {iter_comparison1}")
    print(f"Iteration Comparison2: {iter_comparison2}")

    # np.set_printoptions(threshold=np.inf, precision=2, suppress=True, linewidth=np.inf)
    # print(f"\ngeneral_data:\n", general_data)
    np.savetxt('general_data.txt', general_data, fmt='%.5f')
    return general_data


def sams_calcs():
    # Sam's calcs
    data = {}
    # TS optimization
    lp_file = os.path.join(os.environ["HOME"], "fw_config/sam_launchpad.yaml")
    tag = "sella_prod_1"
    query = {
        "metadata.class": tag
    }
    quacc_data = get_data_wrapper(lp_file, query, collections_name='quacc')

    print('from ts-opt calcs: len(quacc_data):', len(quacc_data))

    os.makedirs('sams_trajectories', exist_ok=True)
    os.makedirs('sams_trajectories/TS', exist_ok=True)
    for doc in quacc_data:
        index = int(doc['name'].split('_')[0][3:])
        traj_array = doc['output']['trajectory']
        niter = len(traj_array)
        data[index] = {'niter_ts': niter}
        os.makedirs(f'sams_trajectories/TS/{index:03}', exist_ok=True)
        try:
            ase.io.write(f'sams_trajectories/TS/{index:03}/{index:03}.xyz',
                         traj_arr_to_atoms_list(traj_array))
        except Exception as e:
            print("TS: an error occurred while accessing trajectory for index:", e)

    # TS-freq
    tag = "sella_prod_freq"
    query = {
        "tags.class": tag
    }
    quacc_data = get_data_wrapper(lp_file, query, collections_name='new_tasks')

    print('from ts-freq calcs: len(quacc_data):', len(quacc_data))

    count = 0
    for doc in quacc_data:
        index = int(doc['task_label'].split('_')[0][3:])
        freq = doc['output']['frequencies'][0]
        electronic_energy = doc['output']['final_energy']
        enthalpy = doc['output']['enthalpy']
        entropy = doc['output']['entropy']
        temperature = 298.15
        gibbs_free_energy = electronic_energy * 27.21139 + 0.0433641 * enthalpy - temperature * entropy * 0.0000433641
        if index in data.keys():
            data[index]['gibbs_free_energy_ts'] = gibbs_free_energy
            data[index]['freq'] = freq
            count += 1
        else:
            print(f"skipping gibbs free energy for index: {index}")

    # quasi-IRC
    # tag = "sella_prod_qirc"
    tag = "sella_prod_irc"
    query = {"metadata.class": tag}
    quacc_data = get_data_wrapper(lp_file, query, collections_name='quacc')
    print('from qirc calcs: len(quacc_data):', len(quacc_data))

    for doc in quacc_data:
        index = int(doc['name'].split('_')[0][3:])
        irc_type = doc['name'].split('_')[2]
        if irc_type == 'forward':
            niter = len(doc['output']['trajectory_results'])
            data[index]['irc_iter_f'] = niter
        elif irc_type == 'reverse':
            niter = len(doc['output']['trajectory_results'])
            data[index]['irc_iter_r'] = niter

    # freq quasi-IRC
    # tag = "sella_prod_qirc_freq"
    tag = "sella_irc_freq"
    query = {'tags.class': tag}
    query_data = get_data_wrapper(lp_file, query, collections_name='new_tasks')
    print('from qirc-freq calcs: len(quacc_data):', len(quacc_data))

    for doc in query_data:
        index = int(doc['task_label'].split('_')[0][3:])
        irc_type = doc['task_label'].split('_')[2]

        # print(f'index: {index}, irc_type: {irc_type}')

        electronic_energy: float = doc['output']['final_energy']
        enthalpy: float = doc['output']['enthalpy']
        entropy: float = doc['output']['entropy']
        temperature: float = 298.15

        gibbs_free_energy = electronic_energy * 27.21139 + enthalpy * 0.0433641 - temperature * entropy * 0.0000433641

        if irc_type == 'forward':
            data[index]['gibbs_free_energy_f'] = gibbs_free_energy
            data[index]['delta_g_f'] = data[index]['gibbs_free_energy_ts'] - data[index]['gibbs_free_energy_f']

        if irc_type == 'reverse':
            data[index]['gibbs_free_energy_r'] = gibbs_free_energy
            data[index]['delta_g_r'] = data[index]['gibbs_free_energy_ts'] - data[index]['gibbs_free_energy_r']
    return data


if __name__ == "__main__":
    # general_data = main()
    # print('general_data:\n', general_data)
    general_data = np.loadtxt('general_data.txt')

    data = sams_calcs()

    count = 0
    count2 = 0
    for element in data:
        data_row_list = list(data[element].values())
        if len(data_row_list) == 9:
            data_row_list.insert(0, element)
            if abs(data_row_list[7] - data_row_list[9]) < 0.01:
                count += 1
            elif 0.01 < abs(data_row_list[7] - data_row_list[9]) < 0.1:
                count2 += 1
    print('\nDelta G forward = Delta G reverse (No reaction) count:', count)
    print('count2:', count2)

    # Extract values from the inner dictionaries and convert to a list of lists
    result_list = []
    for index, inner_dict in data.items():
        # print(index, inner_dict)
        if len(inner_dict) == 9:
            result_list.append([index] + list(inner_dict.values()))
    result_array = np.asarray(result_list)
    np.savetxt('sams_results.txt', result_array, fmt='%.5f')

    count = 0
    for index1 in data.keys():
        for ii, index2 in enumerate(general_data[:, 0]):
            if index1 == index2:
                if general_data[ii, 3]:
                    try:
                        print(data[index1]['delta_g_f'], general_data[ii, 7])
                        if abs(data[index1]['delta_g_f'] - general_data[ii, 7]) > 0.5:
                            count += 1
                            # print(f'index {index1:03} has different results for delta G forward')
                    except Exception as e:
                        print(f'Index {index1} is missing across NewtonNet and Q-CHEM')
    print('count', count)
