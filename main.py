import json
import os
import ase.io
from ase import Atoms
import numpy as np
from numpy import ndarray

from utils import compare_mols, get_data
from typing import List, Dict, Set
from visuals import process_trajectories


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
    general_data: np.ndarray[float] = np.zeros((len(good_indices), 13))

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
                                   delta_g0_r, delta_g1_r, abs(delta_g0_r - delta_g1_r)]
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
                                   delta_g0_r, delta_g1_r, abs(delta_g0_r - delta_g1_r)]

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
                                       delta_g0_r, delta_g1_r, abs(delta_g0_r - delta_g1_r)]

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
                                       delta_g0_r, delta_g1_r, abs(delta_g0_r - delta_g1_r)]

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
                                       delta_g0_r, delta_g1_r, abs(delta_g0_r - delta_g1_r)]
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
                        process_trajectories(os.get_cwd())
                    except Exception as e:
                        print("IRC: an error occurred while accessing trajectory for index:", e)
                os.chdir('../')
            os.chdir('../')
        os.chdir('../')
    os.chdir('../')


def main():
    lp_file = os.path.join(os.environ["HOME"], "fw_config/my_launchpad.yaml")
    # tag = "sella_ts_prod_jun25_[10]"
    tag = "sella_ts_prod_jul2b_[10]"

    # Modify the indices based on your requirements
    indices = np.arange(265)

    # Modify the threshold values based on your requirements
    imag_freq_threshold = 10
    delta_g_threshold = 0.0285

    master_dict = retrieve_data(lp_file, tag, indices)
    log_trajectories(indices, master_dict)
    good_indices = check_present_indices(master_dict, indices)

    set_no_rxn0, set_no_rxn1, iter_comparison1, iter_comparison2, set_same_rxn, set_diff_rxn, set_imag_freqs,\
        set_delta_g_f, set_delta_g_r, general_data = perform_comparisons(master_dict,
                                                                         good_indices,
                                                                         imag_freq_threshold,
                                                                         delta_g_threshold)

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

    np.set_printoptions(threshold=np.inf, precision=2, suppress=True, linewidth=np.inf)
    print(f"\ngeneral_data:\n", general_data)


'''
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
    for doc in quacc_data:
        index = int(doc['name'].split('_')[0][3:])
        energy = doc['output']['trajectory_results'][-1]['energy']
        niter = len(doc['output']['trajectory_results'])
        # print(f"index: {index}, niter: {niter}")
        data[index] = {'niter': niter}

    # TS-freq
    tag = "sella_prod_freq"
    query = {
        "tags.class": tag
    }
    quacc_data = get_data_wrapper(lp_file, query, collections_name='new_tasks')
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
            # data[index]['gibbs_free_energy_ts'] = gibbs_free_energy
            data[index]['freq'] = freq
            count += 1
        else:
            print(f"skipping gibbs free energy for index: {index}")

    # quasi-IRC
    tag = "sella_prod_qirc"
    query = {
        "metadata.class": tag
    }
    quacc_data = get_data_wrapper(lp_file, query, collections_name='quacc')

    for doc in quacc_data:
        index = int(doc['name'].split('_')[0][3:])
        niter = len(doc['output']['trajectory_results'])
        data[index]['irc_iter'] = niter
    for val in data.keys():
        print(val, data[val])
'''


if __name__ == "__main__":
    main()
    # sams_calcs()
