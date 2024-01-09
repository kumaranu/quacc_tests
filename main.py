import json
import os
import ase.io
from ase import Atoms
import numpy as np
from numpy import ndarray
import seaborn as sns
import matplotlib.pyplot as plt
import logging

from utils import compare_mols, get_data
from typing import List, Dict, Set
from visuals import process_trajectories
from utils import get_data_wrapper
from typing import Dict, List, Any


def retrieve_data(lp_file: str, tag: str, indices: List[int]) -> Dict[int, Dict[str, Dict[str, Any]]]:
    """
    Retrieve data from a log file for specified indices and classification tag.

    Parameters
    ----------
    lp_file : str
        The path to the log file.
    tag : str
        The classification tag for filtering data.
    indices : List[int]
        The list of indices for which data needs to be retrieved.

    Returns
    -------
    master_dict : Dict[int, Dict[str, Dict[str, Any]]]
        A nested dictionary containing the retrieved data organized by type and job.

    Notes
    -----
    This function retrieves data from the log file for specified indices, classification tag, and job types.
    It returns a nested dictionary where the outer key represents the TS type (0 or 1), the middle key
    represents the job type ('TS', 'firc', 'rirc'), and the inner dictionary contains the actual data.

    Examples
    --------
    >>> lp_file = 'path/to/log_file.log'
    >>> tag = 'some_classification'
    >>> indices = [1, 2, 3]
    >>> data = retrieve_data(lp_file, tag, indices)
    """
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


def check_present_indices(master_dict: Dict[int, Dict[str, Dict[str, Any]]], indices: List[int]) -> List[int]:
    """
    Check the presence of indices in the master dictionary and filter the valid ones.

    Parameters
    ----------
    master_dict : Dict[int, Dict[str, Dict[str, Any]]]
        The master dictionary containing the data organized by type and job.
    indices : List[int]
        The list of indices to be checked.

    Returns
    -------
    good_indices : List[int]
        The list of indices that are present for all TS types and calculation types.

    Notes
    -----
    This function checks the presence of indices in the master dictionary and filters the valid ones.
    It prints information about failed calculations for each TS type and both types.

    Examples
    --------
    >>> master_dict = {0: {"TS": {...}, "firc": {...}, "rirc": {...}},
    ...                1: {"TS": {...}, "firc": {...}, "rirc": {...}}}
    >>> indices = [1, 2, 3]
    >>> valid_indices = check_present_indices(master_dict, indices)
    """
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
    general_data: np.ndarray[float] = np.zeros((len(good_indices), 23))

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

        ts_energy0 = np.min(master_dict[0]["TS"][index]["energy"])
        ts_energy1 = np.min(master_dict[1]["TS"][index]["energy"])

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
            iter0 = master_dict[0]["TS"][index]["n_iters1"]
            iter1 = master_dict[1]["TS"][index]["n_iters1"]
            iter0b = master_dict[0]["TS"][index]["n_iters2"]
            iter1b = master_dict[1]["TS"][index]["n_iters2"]
            general_data[ii, :] = [index,
                                   check0f0r,
                                   check1f1r,
                                   (check0f1f and check0r1r) or (check0f1r and check1f0r),
                                   imag_freq0, imag_freq1, abs(imag_freq0 - imag_freq1),
                                   delta_g0_f, delta_g1_f, abs(delta_g0_f - delta_g1_f),
                                   delta_g0_r, delta_g1_r, abs(delta_g0_r - delta_g1_r),
                                   e_std_min, e_std_max, e_std_avg, e_std_last,
                                   iter0, iter1, iter0b, iter1b, ts_energy0, ts_energy1]
        # Reactant and product have same bonding for type 1
        if check1f1r:
            set_no_rxn1.add(index)

            # Forward and reverse energy barriers for type 0
            delta_g0_f = gibbs_ts0 - gibbs_f0
            delta_g0_r = gibbs_ts0 - gibbs_r0

            delta_g1_f = gibbs_ts1 - gibbs_f1
            delta_g1_r = gibbs_ts1 - gibbs_r1
            iter0 = master_dict[0]["TS"][index]["n_iters1"]
            iter1 = master_dict[1]["TS"][index]["n_iters1"]
            iter0b = master_dict[0]["TS"][index]["n_iters2"]
            iter1b = master_dict[1]["TS"][index]["n_iters2"]
            general_data[ii, :] = [index,
                                   check0f0r,
                                   check1f1r,
                                   (check0f1f and check0r1r) or (check0f1r and check1f0r),
                                   imag_freq0, imag_freq1, abs(imag_freq0 - imag_freq1),
                                   delta_g0_f, delta_g1_f, abs(delta_g0_f - delta_g1_f),
                                   delta_g0_r, delta_g1_r, abs(delta_g0_r - delta_g1_r),
                                   e_std_min, e_std_max, e_std_avg, e_std_last,
                                   iter0, iter1, iter0b, iter1b, ts_energy0, ts_energy1]

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

                iter0 = master_dict[0]["TS"][index]["n_iters1"]
                iter1 = master_dict[1]["TS"][index]["n_iters1"]
                iter0b = master_dict[0]["TS"][index]["n_iters2"]
                iter1b = master_dict[1]["TS"][index]["n_iters2"]

                general_data[ii, :] = [index,
                                       check0f0r,
                                       check1f1r,
                                       (check0f1f and check0r1r) or (check0f1r and check1f0r),
                                       imag_freq0, imag_freq1, abs(imag_freq0 - imag_freq1),
                                       delta_g0_f, delta_g1_f, abs(delta_g0_f - delta_g1_f),
                                       delta_g0_r, delta_g1_r, abs(delta_g0_r - delta_g1_r),
                                       e_std_min, e_std_max, e_std_avg, e_std_last,
                                       iter0, iter1, iter0b, iter1b, ts_energy0, ts_energy1]

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

                iter0 = master_dict[0]["TS"][index]["n_iters1"]
                iter1 = master_dict[1]["TS"][index]["n_iters1"]
                iter0b = master_dict[0]["TS"][index]["n_iters2"]
                iter1b = master_dict[1]["TS"][index]["n_iters2"]
                general_data[ii, :] = [index,
                                       check0f0r,
                                       check1f1r,
                                       (check0f1f and check0r1r) or (check0f1r and check1f0r),
                                       imag_freq0, imag_freq1, abs(imag_freq0 - imag_freq1),
                                       delta_g0_f, delta_g1_f, abs(delta_g0_f - delta_g1_f),
                                       delta_g0_r, delta_g1_r, abs(delta_g0_r - delta_g1_r),
                                       e_std_min, e_std_max, e_std_avg, e_std_last,
                                       iter0, iter1, iter0b, iter1b, ts_energy0, ts_energy1]

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

                iter0 = master_dict[0]["TS"][index]["n_iters1"]
                iter1 = master_dict[1]["TS"][index]["n_iters1"]
                iter0b = master_dict[0]["TS"][index]["n_iters2"]
                iter1b = master_dict[1]["TS"][index]["n_iters2"]
                general_data[ii, :] = [index,
                                       check0f0r,
                                       check1f1r,
                                       (check0f1f and check0r1r) or (check0f1r and check1f0r),
                                       imag_freq0, imag_freq1, abs(imag_freq0 - imag_freq1),
                                       delta_g0_f, delta_g1_f, abs(delta_g0_f - delta_g1_f),
                                       delta_g0_r, delta_g1_r, abs(delta_g0_r - delta_g1_r),
                                       e_std_min, e_std_max, e_std_avg, e_std_last,
                                       iter0, iter1, iter0b, iter1b, ts_energy0, ts_energy1]
    return set_no_rxn0, set_no_rxn1, iter_comparison1, iter_comparison2, set_same_rxn, set_diff_rxn,\
        set_imag_freqs, set_delta_g_f, set_delta_g_r, general_data


def traj_arr_to_atoms_list(traj_array):
    """
    Convert a trajectory array to a list of ASE Atoms objects.

    This function takes a trajectory array, where each entry contains information
    about atoms' numbers and positions, and converts it into a list of ASE Atoms objects.

    Parameters:
    traj_array (list): A list of trajectory entries, each containing atoms' numbers and positions.

    Returns:
    list: A list of ASE Atoms objects representing the trajectory.
    """
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
    """
    Log transition state information.

    This function logs transition state information from the master_dict into
    XYZ format files.

    Parameters:
    indices (list): A list of indices corresponding to the transition states to log.
    master_dict (dict): A dictionary containing the transition state data.

    Returns:
    None
    """
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
    """
    Log trajectory information.

    This function logs trajectory information from the master_dict into XYZ format files.

    Parameters:
    indices (list): A list of indices corresponding to the trajectories to log.
    master_dict (dict): A dictionary containing the trajectory data.

    Returns:
    None
    """
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
                      columns_to_compare=None,
                      row_labels=None):
    """
    Plot a heatmap of the correlation matrix.

    This function calculates the correlation matrix for the specified columns in the
    given general_data array and plots a heatmap.

    Parameters:
    general_data (numpy.ndarray): A 2D array containing the data for which to calculate correlations.
    columns_to_compare (list or None, optional): A list of column indices to include in correlation calculation.
        If None, default columns [1, 2, 3, 6, 9, 12, 16] will be used.

    Returns:
    None
    """
    # Calculate correlation matrix
    if columns_to_compare is None:
        columns_to_compare = [1, 2, 3, 6, 9, 12, 16]
    correlation_matrix = np.corrcoef(general_data[:, columns_to_compare],
                                     rowvar=False)
    # print('correlation_matrix:\n', correlation_matrix)

    if row_labels is None:
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
    # Configure the logging settings
    logging.basicConfig(filename='log.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        lp_file = os.path.join(os.environ["HOME"], "fw_config/my_launchpad.yaml")
        # tag = "sella_ts_prod_jun25_[10]"
        # tag = "sella_ts_prod_jul2b_[10]"
        tag = "sella_ts_prod_jul13d_[10]"
        # tag = "sella_ts_prod_aug1_[10]"
        # tag = "sella_ts_prod_aug1b_[10]"
        # tag = "sella_ts_prod_oct4_[10]"

        # Modify the indices based on your requirements
        indices = np.arange(265)
        # indices = [24, 38, 71, 107, 112, 167, 185, 189, 218, 237, 240, 253, 256]
        # indices = [107, 185]

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

        logging.info(f"Number of reactions with no reaction in 0: {len(set_no_rxn0)}")
        logging.info(f"Number of reactions with no reaction in 1: {len(set_no_rxn1)}")
        logging.info(f"Number of reactions with no reaction in both 0 and 1: {len(set_no_rxn0.intersection(set_no_rxn1))}")

        logging.info(f"Number of reactions with the same reaction in both 0 and 1: {len(set_same_rxn)}")
        logging.info(f"Number of reactions with different reaction in both 0 and 1: {len(set_diff_rxn)}")

        logging.info(f"Number of different imaginary frequency numbers: {len(set_imag_freqs)}")
        logging.info(f"Number of different DeltaG (forward) numbers: {len(set_delta_g_f)}")
        logging.info(f"Number of different DeltaG (reverse) numbers: {len(set_delta_g_r)}")

        logging.info(f"Iteration Comparison1: {iter_comparison1}")
        logging.info(f"Iteration Comparison2: {iter_comparison2}")
        np.savetxt('general_data.txt', general_data, fmt='%.5f')
        logging.info("General data saved to 'general_data.txt'")
        '''
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
        '''
        return general_data
    except Exception as e:
        logging.error(f"An error occured: {e}")
        raise


def sams_calcs():
    qchem_dict = {f'{i:03}':{} for i in range(265)}
    # Sam's calcs
    data = {}
    # TS optimization
    lp_file = os.path.join(os.environ["HOME"], "fw_config/sam_launchpad.yaml")
    query = {"metadata.class": "sella_prod_1"}
    quacc_data1 = get_data_wrapper(lp_file, query, collections_name='quacc')
    # print('type(quacc_data):', type(quacc_data1))
    # print(quacc_data1[0].keys())
    # print(quacc_data1[0]['metadata'].keys())
    # print(quacc_data1[0]['output'].keys())
    # print(quacc_data1[0]['output']['name'])
    # print('from ts-opt calcs: len(quacc_data):', len(quacc_data1))

    os.makedirs('sams_trajectories', exist_ok=True)
    os.makedirs('sams_trajectories/TS', exist_ok=True)
    for doc in quacc_data1:
        index = int(doc['name'].split('_')[0][3:])
        qchem_dict[f'{index:03}']['ts'] = doc
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
    query2 = {"tags.class": "sella_prod_freq"}
    quacc_data2 = get_data_wrapper(lp_file, query2, collections_name='new_tasks')
    print('from ts-freq calcs: len(quacc_data):', len(quacc_data2))

    count = 0
    for doc in quacc_data2:
        index = int(doc['task_label'].split('_')[0][3:])
        qchem_dict[f'{index:03}']['ts_freq'] = doc
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
    query3 = {"metadata.class": "sella_prod_irc"}
    quacc_data3 = get_data_wrapper(lp_file, query3, collections_name='quacc')
    print('from qirc calcs: len(quacc_data):', len(quacc_data3))

    for doc in quacc_data3:
        index = int(doc['name'].split('_')[0][3:])
        irc_type = doc['name'].split('_')[2]
        if irc_type == 'forward':
            qchem_dict[f'{index:03}']['firc'] = doc
            niter = len(doc['output']['trajectory_results'])
            data[index]['irc_iter_f'] = niter
        elif irc_type == 'reverse':
            qchem_dict[f'{index:03}']['rirc'] = doc
            niter = len(doc['output']['trajectory_results'])
            data[index]['irc_iter_r'] = niter

    # freq quasi-IRC
    # tag = "sella_prod_qirc_freq"
    tag = "sella_irc_freq"
    query4 = {'tags.class': tag}
    quacc_data4 = get_data_wrapper(lp_file, query4, collections_name='new_tasks')
    print('from qirc-freq calcs: len(quacc_data):', len(quacc_data4))

    for doc in quacc_data4:
        index = int(doc['task_label'].split('_')[0][3:])
        irc_type = doc['task_label'].split('_')[2]

        # print(f'index: {index}, irc_type: {irc_type}')

        electronic_energy: float = doc['output']['final_energy']
        enthalpy: float = doc['output']['enthalpy']
        entropy: float = doc['output']['entropy']
        temperature: float = 298.15

        gibbs_free_energy = electronic_energy * 27.21139 + enthalpy * 0.0433641 - temperature * entropy * 0.0000433641

        if irc_type == 'forward':
            qchem_dict[f'{index:03}']['firc_freq'] = doc
            data[index]['gibbs_free_energy_f'] = gibbs_free_energy
            data[index]['delta_g_f'] = data[index]['gibbs_free_energy_ts'] - data[index]['gibbs_free_energy_f']

        if irc_type == 'reverse':
            qchem_dict[f'{index:03}']['rirc_freq'] = doc
            data[index]['gibbs_free_energy_r'] = gibbs_free_energy
            data[index]['delta_g_r'] = data[index]['gibbs_free_energy_ts'] - data[index]['gibbs_free_energy_r']

    print(len([key for key, value in qchem_dict.items() if len(value) > 5]))

    return data

def comparison_nn_dft(general_data: np.ndarray, data: Dict[str, Dict[str, float]]) -> None:
    """
    Compare data from a general dataset (NewtonNet) and a specific dataset (DFT).

    Parameters
    ----------
    general_data : numpy.ndarray
        The general dataset (NewtonNet) organized as a NumPy array.
    data : Dict[str, Dict[str, float]]
        The specific dataset (DFT) organized as a dictionary.

    Returns
    -------
    None
        Prints information about the comparison results.

    Notes
    -----
    This function compares data from a general dataset (NewtonNet) and a specific dataset (DFT).
    It counts cases where Delta G forward is equal or within a specific range of Delta G reverse.
    It also prints information about cases where the datasets have different results for Delta G forward.

    Examples
    --------
    >>> general_data = np.array([[...], [...], ...])
    >>> data = {'element1': {'property1': value1, 'property2': value2, ...}, ...}
    >>> comparison_nn_dft(general_data, data)
    """
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


'''
def graph_compare(general_data, data, good_indices):
    for ii, index in enumerate(good_indices):
        check0f0r = compare_mols(, )
        check1f1r = compare_mols(, )
        check0f1f = compare_mols(, )
        check0r1r = compare_mols(, )
        check0f1r = compare_mols(, )
        check1f0r = compare_mols(, )

        if check0f0r:
            set_no_rxn0.add(index)
            general_data[ii, :] = [index, check0f0r, check1f1r, (check0f1f and check0r1r) or (check0f1r and check1f0r)]
        if check1f1r:
            set_no_rxn1.add(index)
            general_data[ii, :] = [index, check0f0r, check1f1r, (check0f1f and check0r1r) or (check0f1r and check1f0r)]

        if not check0f0r and not check1f1r:
            if check0f1f and check0r1r:
                set_same_rxn.add(index)
                general_data[ii, :] = [index, check0f0r, check1f1r, (check0f1f and check0r1r) or (check0f1r and check1f0r)]
            elif check0f1r and check1f0r:
                set_same_rxn.add(index)
                general_data[ii, :] = [index, check0f0r, check1f1r, (check0f1f and check0r1r) or (check0f1r and check1f0r)]
            else:
                set_diff_rxn.add(index)
                general_data[ii, :] = [index, check0f0r, check1f1r, (check0f1f and check0r1r) or (check0f1r and check1f0r)]
'''


def nn_rxn_plot(x):
    # Extract the columns
    column1 = x[:, 1]
    column2 = x[:, 2]
    column3 = x[:, 3]

    # Calculate the sums
    type0_rxn = int(np.sum(column1))
    type1_rxn = int(np.sum(column2))
    both_same_rxn = int(np.sum(column1 * column2 * column3))
    both_diff_rxn = int(np.sum((1-column1) * (1-column2) * (1-column3)))

    # Create a bar chart
    plt.figure(figsize=(10, 8))
    bars = plt.bar(['Using\n ML-Hessian', 'Without\n Hessian', 'Both\n cases'],
                   [type0_rxn, type1_rxn, both_same_rxn, both_diff_rxn],
                   color=['blue', 'green', 'red', 'orange'],
                   width=0.75
                   )

    # Add labels to the bars
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center',
                     va='bottom',
                     fontsize=22)

    # Add labels and title
    # plt.xlabel('Type of calculation', fontsize=24)
    plt.ylabel('Number of tests', fontsize=24)
    plt.title('No chemical reactions',
              fontsize=26)

    # Adjust font size for tick labels on both x and y axes
    plt.xticks(fontsize=22)  # Increase fontsize here for x-axis tick labels
    plt.yticks(fontsize=22)  # Increase fontsize here for y-axis tick labels

    # Set y-axis limits to provide space for the title
    plt.ylim(top=max(type0_rxn, type1_rxn, both_same_rxn, both_diff_rxn) * 1.1)

    # Show the plot
    plt.tight_layout()
    plt.show()
    # plt.savefig('rxn.png', dpi=300)


def nn_no_rxn_plot(x):
    # Extract the columns
    column1 = x[:, 1]
    column2 = x[:, 2]

    # Calculate the sums
    sum_column1 = int(np.sum(column1))
    sum_column2 = int(np.sum(column2))
    sum_product = int(np.sum(column1 * column2))

    # Create a bar chart
    plt.figure(figsize=(10, 8))
    bars = plt.bar(['Using\n ML-Hessian', 'Without\n Hessian', 'Both\n cases'],
                   [sum_column1, sum_column2, sum_product],
                   color=['blue', 'green', 'red'],
                   width=0.75
                   )

    # Add labels to the bars
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center',
                     va='bottom',
                     fontsize=22)

    # Add labels and title
    # plt.xlabel('Type of calculation', fontsize=24)
    plt.ylabel('Number of tests', fontsize=24)
    plt.title('No chemical reactions',
              fontsize=26)

    # Adjust font size for tick labels on both x and y axes
    plt.xticks(fontsize=22)  # Increase fontsize here for x-axis tick labels
    plt.yticks(fontsize=22)  # Increase fontsize here for y-axis tick labels

    # Set y-axis limits to provide space for the title
    plt.ylim(top=max(sum_column1, sum_column2, sum_product) * 1.1)  # Increase or decrease 1.2 to control space

    # Show the plot
    plt.tight_layout()
    # plt.show()
    plt.savefig('no_rxn.png', dpi=300)


def nn_diff_property_plot(x):
    # Extract the columns
    column1 = x[:, 1]
    column2 = x[:, 2]

    # Calculate the sums
    sum_column1 = int(np.sum((1 - x[:, 1]) * (1 - x[:, 2]) * x[:, 3] * (x[:, 6] > 10)))
    sum_column2 = int(np.sum((1 - x[:, 1]) * (1 - x[:, 2]) * x[:, 3] * (x[:, 9] > 0.0285)))
    sum_product = int(np.sum((1 - x[:, 1]) * (1 - x[:, 2]) * x[:, 3] * (x[:, 12] > 0.0285)))

    # Create a bar chart
    plt.figure(figsize=(10, 8))
    bars = plt.bar(
        [
            '$\Delta$' + r'$\nu$' + '$> 10\iota$' + '\n' + r'$\left(cm^{-1}\right)$',
            r'$\Delta G^{\ddag}_{forward} > 10$' + '\n' + r'$\left(cm^{-1}\right)$',
            r'$\Delta G^{\ddag}_{reverse} > 10$' + '\n' + r'$\left(cm^{-1}\right)$',
        ],
        [sum_column1, sum_column2, sum_product],
        color=['blue', 'green', 'red'],
        width=0.75
    )

    # Add labels to the bars
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center',
                     va='bottom',
                     fontsize=22)

    # Add labels and title
    # plt.xlabel('Type of calculation', fontsize=24)
    plt.ylabel('Number of tests', fontsize=24)
    plt.title('Difference in the properties for\n with/without Hessian results',
              fontsize=26)

    # Adjust font size for tick labels on both x and y axes
    plt.xticks(fontsize=22)  # Increase fontsize here for x-axis tick labels
    plt.yticks(fontsize=22)  # Increase fontsize here for y-axis tick labels

    # Set y-axis limits to provide space for the title
    plt.ylim(top=max(sum_column1, sum_column2, sum_product) * 1.1)  # Increase or decrease 1.2 to control space

    # Show the plot
    plt.tight_layout()
    # plt.show()
    plt.savefig('diff_property.png', dpi=300)


if __name__ == "__main__":
    general_data = main()
    print('general_data:\n', general_data)
    general_data = np.loadtxt('general_data.txt')

    # nn_no_rxn_plot(general_data)
    # nn_rxn_plot(general_data) ERROR IN THIS FUNCTION
    # nn_diff_property_plot(general_data)

    # data = sams_calcs()

    # comparison_nn_dft(general_data, data)
