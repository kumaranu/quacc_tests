import os
import numpy as np
from utils import compare_mols, get_data


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
    count0, count1, count_10 = 0, 0, 0
    set_err_0, set_err_1, set_err_both = set(), set(), set()
    good_indices = []
    for index in indices:
        all_present = {0: True, 1: True}
        for ts_type in [0, 1]:
            for calc_type in ["TS", "firc", "rirc"]:
                if index not in master_dict[ts_type][calc_type]:
                    print("index, ts_type, calc_type:", index, ts_type, calc_type)
                    if ts_type == 0:
                        set_err_0.add(index)
                        count0 += 1
                    elif ts_type == 1:
                        set_err_1.add(index)
                        count1 += 1
                    all_present[ts_type] = False
        if all(all_present.values()):
            good_indices.append(index)
    print(f'count0, count1: {count0}, {count1}')
    print(f'Only 0 failed, {len(set_err_0-set_err_1)}: {set_err_0 - set_err_1}')
    print(f'Only 1 failed:', set_err_1-set_err_0)
    print('Both 0 and 1 failed', set_err_1.intersection(set_err_0))
    print('len(good_indices):', len(good_indices))
    return good_indices


def perform_comparisons(master_dict, good_indices, imag_freq_threshold, delta_g_threshold):
    same_ts_indices = []
    iso_checks = {0: 0, 1: 0}
    iter_comparison = {0: 0, 1: 0}
    same_ts = 0
    diff_imag_freq_nums = 0
    diff_delta_g_f_nums = 0
    diff_delta_g_r_nums = 0
    set_succeed0, set_succeed1 = set(), set()

    for index in good_indices:
        check0f0r = compare_mols(master_dict[0]["firc"][index]["mol"], master_dict[0]["rirc"][index]["mol"])
        check1f1r = compare_mols(master_dict[1]["firc"][index]["mol"], master_dict[1]["rirc"][index]["mol"])
        check0f1f = compare_mols(master_dict[0]["firc"][index]["mol"], master_dict[1]["firc"][index]["mol"])
        check0r1r = compare_mols(master_dict[0]["rirc"][index]["mol"], master_dict[1]["rirc"][index]["mol"])
        check0f1r = compare_mols(master_dict[0]["firc"][index]["mol"], master_dict[1]["rirc"][index]["mol"])
        check1f0r = compare_mols(master_dict[1]["firc"][index]["mol"], master_dict[0]["rirc"][index]["mol"])

        if not check0f0r:
            iso_checks[0] += 1
            set_succeed0.add(index)
        if not check1f1r:
            iso_checks[1] += 1
            set_succeed1.add(index)

        if not check0f0r and not check1f1r:
            if check0f1f and check0r1r:
                same_ts += 1
                iter_comparison[0] += master_dict[0]["TS"][index]["n_iters"]
                iter_comparison[1] += master_dict[1]["TS"][index]["n_iters"]
                same_ts_indices.append(index)
                imag_freq0 = np.min(master_dict[0]["TS"][index]["imag_vib_freq"])
                imag_freq1 = np.min(master_dict[1]["TS"][index]["imag_vib_freq"])

                if abs(imag_freq0 - imag_freq1) > imag_freq_threshold:
                    diff_imag_freq_nums += 1

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

                if abs(delta_g0_f - delta_g1_f) > delta_g_threshold:
                    diff_delta_g_f_nums += 1
                if abs(delta_g0_r - delta_g1_r) > delta_g_threshold:
                    diff_delta_g_r_nums += 1

            elif check0f1r and check1f0r:
                same_ts += 1
                iter_comparison[0] += master_dict[0]["TS"][index]["n_iters"]
                iter_comparison[1] += master_dict[1]["TS"][index]["n_iters"]
                same_ts_indices.append(index)
                imag_freq0 = np.min(master_dict[0]["TS"][index]["imag_vib_freq"])
                imag_freq1 = np.min(master_dict[1]["TS"][index]["imag_vib_freq"])

                if abs(imag_freq0 - imag_freq1) > imag_freq_threshold:
                    diff_imag_freq_nums += 1

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
                    diff_delta_g_f_nums += 1
                if abs(delta_g0_r - delta_g1_r) > delta_g_threshold:
                    diff_delta_g_r_nums += 1

    return same_ts_indices, iso_checks, set_succeed0, set_succeed1, iter_comparison, same_ts, diff_imag_freq_nums, diff_delta_g_f_nums,\
        diff_delta_g_r_nums


def main():
    lp_file = os.path.join(os.environ["HOME"], "fw_config/my_launchpad.yaml")
    tag = "sella_ts_prod_jun25_[10]"

    # Modify the indices based on your requirements
    indices = np.arange(265)

    # Modify the threshold values based on your requirements
    imag_freq_threshold = 10
    delta_g_threshold = 0.0285

    master_dict = retrieve_data(lp_file, tag, indices)
    good_indices = check_present_indices(master_dict, indices)

    same_ts_indices, iso_checks, set_succeed0, set_succeed1, iter_comparison, same_ts, diff_imag_freq_nums, diff_delta_g_f_nums,\
        diff_delta_g_r_nums = perform_comparisons(master_dict, good_indices, imag_freq_threshold, delta_g_threshold)

    # print("Same TS Indices: ", same_ts_indices)
    print("Isomer Check: ", iso_checks)
    print("Iteration Comparison: ", iter_comparison)
    print("Same TS: ", same_ts)
    print("Different Imaginary Frequency Numbers: ", diff_imag_freq_nums)
    print("Different DeltaG (forward) Numbers: ", diff_delta_g_f_nums)
    print("Different DeltaG (reverse) Numbers: ", diff_delta_g_r_nums)
    failed_ts_indices0 = set(indices)-set_succeed0
    failed_ts_indices1 = set(indices)-set_succeed1
    failed_ts_indices1
    print('Number of failed TS for type0:', len(failed_ts_indices0))
    print('Number of failed TS for type1:', len(failed_ts_indices1))
    print('Did not find transition state for type 0 (with custom hessian):', failed_ts_indices0)
    print('Did not find transition state for type 1 (without custom hessian):', failed_ts_indices1)
    print('Indices that failed to find TS in both ts_types:', failed_ts_indices1.intersection(failed_ts_indices0))

if __name__ == "__main__":
    main()
