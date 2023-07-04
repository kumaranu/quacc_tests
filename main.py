import os
import numpy as np
from utils import compare_mols, get_data
from typing import List, Dict, Set, Tuple


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
    print(f'Number of all-good indices: {len(good_indices)}\n')
    return good_indices


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
    iter_comparison: Dict[int, int] = {0: 0, 1: 0}
    set_same_ts: Set[int] = set()
    set_imag_freqs: Set[int] = set()
    set_delta_g_f: Set[int] = set()
    set_delta_g_r: Set[int] = set()
    set_failed0: Set[int] = set()
    set_failed1: Set[int] = set()

    for index in good_indices:
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
        delta_g0_f = gibbs_ts0 - gibbs_f0
        delta_g0_r = gibbs_ts0 - gibbs_r0

        delta_g1_f = gibbs_ts1 - gibbs_f1
        delta_g1_r = gibbs_ts1 - gibbs_r1
        if check0f0r:
            imag_freq0 = np.min(master_dict[0]["TS"][index]["imag_vib_freq"])
            set_failed0.add((index, imag_freq0))
        if check1f1r:
            # imag_freq1 = np.min(master_dict[1]["TS"][index]["imag_vib_freq"])
            set_failed1.add((index, imag_freq1))

        if not check0f0r and not check1f1r:
            if check0f1f and check0r1r:
                set_same_ts.add(index)
                iter_comparison[0] += master_dict[0]["TS"][index]["n_iters"]
                iter_comparison[1] += master_dict[1]["TS"][index]["n_iters"]
                imag_freq0 = np.min(master_dict[0]["TS"][index]["imag_vib_freq"])
                imag_freq1 = np.min(master_dict[1]["TS"][index]["imag_vib_freq"])

                if abs(imag_freq0 - imag_freq1) > imag_freq_threshold:
                    set_imag_freqs.add(index)

                if abs(delta_g0_f - delta_g1_f) > delta_g_threshold:
                    set_delta_g_f.add(index)
                if abs(delta_g0_r - delta_g1_r) > delta_g_threshold:
                    set_delta_g_r.add(index)

            elif check0f1r and check1f0r:
                set_same_ts.add(index)
                iter_comparison[0] += master_dict[0]["TS"][index]["n_iters"]
                iter_comparison[1] += master_dict[1]["TS"][index]["n_iters"]
                imag_freq0 = np.min(master_dict[0]["TS"][index]["imag_vib_freq"])
                imag_freq1 = np.min(master_dict[1]["TS"][index]["imag_vib_freq"])

                if abs(imag_freq0 - imag_freq1) > imag_freq_threshold:
                    set_imag_freqs.add(index)

                gibbs_ts0 = master_dict[0]["TS"][index]["gibbs_free_energy"]
                gibbs_f0 = master_dict[0]["TS"][index]["gibbs_free_energy"]
                gibbs_r0 = master_dict[0]["TS"][index]["gibbs_free_energy"]

                gibbs_ts1 = master_dict[1]["TS"][index]["gibbs_free_energy"]
                gibbs_f1 = master_dict[1]["TS"][index]["gibbs_free_energy"]
                gibbs_r1 = master_dict[1]["TS"][index]["gibbs_free_energy"]
                delta_g0_f = gibbs_ts0 - gibbs_f0
                delta_g0_r = gibbs_ts0 - gibbs_r0

                delta_g1_f = gibbs_ts1 - gibbs_f1
                delta_g1_r = gibbs_ts1 - gibbs_r1

                if abs(delta_g0_f - delta_g1_f) > delta_g_threshold:
                    set_delta_g_f.add(index)
                if abs(delta_g0_r - delta_g1_r) > delta_g_threshold:
                    set_delta_g_r.add(index)

    return set_failed0, set_failed1, iter_comparison, set_same_ts, set_imag_freqs, set_delta_g_f, set_delta_g_r


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

    good_indices = check_present_indices(master_dict, indices)

    set_failed0, set_failed1, iter_comparison, set_same_ts, set_imag_freqs, set_delta_g_f,\
        set_delta_g_r = perform_comparisons(master_dict, good_indices, imag_freq_threshold, delta_g_threshold)

    print(f"set TS failed0: {len(set_failed0)}: {set_failed0}")
    print(f"set TS failed1: {len(set_failed1)}: {set_failed1}")
    print(f"set both 1 and 0 TS failed: {len(set_failed0.intersection(set_failed1))}: {set_failed0.intersection(set_failed1)}")
    print(f"\nIteration Comparison: {iter_comparison}")
    print(f"\nset_same_ts: {len(set_same_ts)}: {set_same_ts}")
    print(f"\nDifferent Imaginary Frequency Numbers: {len(set_imag_freqs)}: {set_imag_freqs}")
    print(f"Different DeltaG (forward) Numbers: {len(set_delta_g_f)}: {set_delta_g_f}")
    print(f"Different DeltaG (reverse) Numbers: {len(set_delta_g_r)}: {set_delta_g_r}")


if __name__ == "__main__":
    main()
