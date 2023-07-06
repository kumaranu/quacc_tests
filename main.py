import os
import numpy as np
from utils import compare_mols, get_data
from typing import List, Dict, Set, Tuple
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
    print(f'Calc-fail count0, count1: {len(set_err_0)}, {len(set_err_1)}')
    print(f'Only 0 failed: {len(set_err_0-set_err_1)}: {set_err_0 - set_err_1}')
    print(f'Only 1 failed: {len(set_err_1-set_err_0)}: {set_err_1-set_err_0}')
    print(f'Both 0 and 1 failed: {len(set_err_1.intersection(set_err_0))}:'
          f' {set_err_1.intersection(set_err_0)}')
    print(f'\nNumber of all-good indices: {len(good_indices)}')
    return good_indices

'''
def perform_comparisons(
        master_dict: Dict[int, Dict[str, List[Dict[str, any]]]],
        good_indices: List[int],
        imag_freq_threshold: float,
        delta_g_threshold: float
) -> Tuple[
    Dict[int, int],
    Set[int],
    Set[int],
    Dict[int, int],
    Set[int],
    Set[int],
    Set[int],
    Set[int],
]:
    iter_comparison1: Dict[int, int] = {0: 0, 1: 0}
    iter_comparison2: Dict[int, int] = {0: 0, 1: 0}
    set_same_ts: Set[int] = set()
    set_diff_ts1: Set[int] = set()
    set_diff_ts0: Set[int] = set()
    set_diff_ts10: Set[int] = set()
    set_imag_freqs: Set[int] = set()
    set_delta_g_f: Set[int] = set()
    set_delta_g_r: Set[int] = set()
    set_failed0: Set[int] = set()
    set_failed1: Set[int] = set()
    general_data: np.ndarray[float] = np.zeros((len(good_indices), 7))

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
            set_failed0.add(index)
        # Reactant and product have same bonding for type 1
        if check1f1r:
            set_failed1.add(index)

        # Reactant and product have different bonding for type 0
        if not check0f0r:
            # Reactant and product have different bonding for type 1
            if not check1f1r:

                ###############################################
                #### Everything here is a transition state ####
                ###############################################

                # Imaginary frequency is differing more than a threshold
                if abs(imag_freq0 - imag_freq1) > imag_freq_threshold:
                    set_imag_freqs.add(index)
                # The reactant from type 0 and the reactant from type 1 have the same graph
                if check0f1f and check0r1r:
                        set_same_ts.add(index)

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
                    # Reactance has the same graph but product doesn't
                elif check0f1f and check0r1r:
                    else:
                        print(index, "reactant0 has the same graph as reactant1 but the product doesn't")
                elif check0r1r:
                    # The product from type 0 and the product from type 1 have the same graph
                    if check0f1r:
                        set_same_ts.add(index)

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
                    # Reactance has the same graph but product doesn't
                    else:
                        print(index, "reactant0 has the same graph as reactant1 but the product doesn't")
                # Reactants and products are switched between type 0 and type 1
                elif check0f1r:
                    if check1f0r:
                        set_same_ts.add(index)

                        iter_comparison1[0] += master_dict[0]["TS"][index]["n_iters1"]
                        iter_comparison1[1] += master_dict[1]["TS"][index]["n_iters1"]
                        iter_comparison2[0] += master_dict[0]["TS"][index]["n_iters2"]
                        iter_comparison2[1] += master_dict[1]["TS"][index]["n_iters2"]

                        # Forward and reverse energy barriers for type 0
                        delta_g0_f = gibbs_ts0 - gibbs_f0
                        delta_g0_r = gibbs_ts0 - gibbs_r0

                        ###!!!!!!!############ NOTE ############!!!!!!!###
                        ######## Using reverse IRC for Forward ###########
                        ###################### and #######################
                        ############## forward IRC for reverse ###########
                        ##### to calculate energy barriers for type 1 ####
                        delta_g1_f = gibbs_ts1 - gibbs_r1
                        delta_g1_r = gibbs_ts1 - gibbs_f1
                    else:
                        print(index, 'graph from product1 matches with reactant0 but not the other way.')
                else:
                    print('TS exists in both 0 and 1 but Different TS!!!!')
                    general_data[ii, :] = [index, imag_freq0, imag_freq1, delta_g0_f, delta_g1_f, delta_g0_r,
                                           delta_g1_r]
                if abs(delta_g0_f - delta_g1_f) > delta_g_threshold:
                    set_delta_g_f.add(index)
                if abs(delta_g0_r - delta_g1_r) > delta_g_threshold:
                    set_delta_g_r.add(index)
                general_data[ii, :] = [index, imag_freq0, imag_freq1, delta_g0_f, delta_g1_f, delta_g0_r, delta_g1_r]
            else:
                print('TS failed only in 1')
                set_diff_ts1.add(index)
                general_data[ii, :] = [index, imag_freq0, imag_freq1, delta_g0_f, delta_g1_f, delta_g0_r, delta_g1_r]
        elif check0f0r and not check1f1r:
            print('TS failed only in 0')
            set_diff_ts0.add(index)
        else:
            print('TS failed in both 0 and 1')
            set_diff_ts10.add(index)
    return set_failed0, set_failed1, iter_comparison1, iter_comparison2, set_same_ts, set_diff_ts0, set_diff_ts1,\
        set_diff_ts10, set_imag_freqs, set_delta_g_f, set_delta_g_r, general_data
'''

def main():
    lp_file = os.path.join(os.environ["HOME"], "fw_config/my_launchpad.yaml")
    # tag = "sella_ts_prod_jun25_[10]"
    tag = "sella_ts_prod_jul2b_[10]"

    # Modify the indices based on your requirements
    indices = np.arange(265)

    # Modify the threshold values based on your requirements
    imag_freq_threshold = 10 * 2
    delta_g_threshold = 0.0285 * 2

    master_dict = retrieve_data(lp_file, tag, indices)
    print(master_dict.keys())
    import sys
    sys.exit()
    good_indices = check_present_indices(master_dict, indices)

    set_failed0, set_failed1, iter_comparison1, iter_comparison2, set_same_ts, set_diff_ts0,\
        set_diff_ts1, set_diff_ts10, set_imag_freqs, set_delta_g_f, set_delta_g_r,\
        general_data = perform_comparisons(master_dict,
                                           good_indices,
                                           imag_freq_threshold,
                                           delta_g_threshold)

    print(f"\nset TS failed0: {len(set_failed0)}: {set_failed0}")
    #for item in set_failed0:
    #    print(f"({item[0]:>3d}, {item[1]:>8.2f}, {item[2]:>6.2f}, {item[3]:>6.2f})")
    print(f"set TS failed1: {len(set_failed1)}: {set_failed1}")
    #for item in set_failed1:
    #    print(f"({item[0]:>3d}, {item[1]:>8.2f}, {item[2]:>6.2f}, {item[3]:>6.2f})")
    print(f"set both 1 and 0 TS failed: {len(set_failed0.intersection(set_failed1))}:"
          f" {set_failed0.intersection(set_failed1)}")

    print(f"\nIteration Comparison1: {iter_comparison1}")
    print(f"Iteration Comparison2: {iter_comparison2}")

    print(f"\nset_same_ts: {len(set_same_ts)}: {set_same_ts}")
    print(f"set_diff_ts: {len(set_diff_ts)}: {set_diff_ts}")

    print(f"\nDifferent Imaginary Frequency Numbers: {len(set_imag_freqs)}: {set_imag_freqs}")
    print(f"Different DeltaG (forward) Numbers: {len(set_delta_g_f)}: {set_delta_g_f}")
    print(f"Different DeltaG (reverse) Numbers: {len(set_delta_g_r)}: {set_delta_g_r}")

    np.set_printoptions(threshold=np.inf, precision=2, suppress=True)
    print(f"general_data:\n", general_data)

def sams_calcs():
    #Sam's calcs
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
        # print(f'index: {index}, niter: {niter}')
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
            print(f'skipping gibbs free energy for index: {index}')

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


if __name__ == "__main__":
    main()
    #sams_calcs()
