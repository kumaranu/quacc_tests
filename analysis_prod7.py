import os
import numpy as np
from graphs_analysis import plot_graph_comps
from utils import get_data_wrapper, compare_mols, clean_calcs, get_data

lp_file = os.path.join(os.environ["HOME"], "fw_config/my_launchpad.yaml")
tag = "sella_ts_prod_jun21b_[10]"
threshold = -3.6
indices = np.arange(265)

# Retrieve data
dd_TS, dd_firc, dd_rirc = {}, {}, {}
for ts_type in [0, 1]:
    dd_TS[ts_type],   _, _ = get_data(indices, lp_file, class_tag=tag, ts_type=ts_type, job_type='TS')
    dd_firc[ts_type], _, _ = get_data(indices, lp_file, class_tag=tag, ts_type=ts_type, job_type='irc-forward')
    dd_rirc[ts_type], _, _ = get_data(indices, lp_file, class_tag=tag, ts_type=ts_type, job_type='irc-reverse')

print('dd_TS:', dd_TS)

# Create a dictionary to store data
master_dict = {0: {"TS": dd_TS[0], "firc": dd_firc[0], "rirc": dd_rirc[0]},
               1: {"TS": dd_TS[1], "firc": dd_firc[1], "rirc": dd_rirc[1]}}

# Initialize counters and lists
num_all_type_present = {0: 0, 1: 0}
num_all_present = 0
good_indices = []
iso_checks = {0: 0, 1: 0}
iter_comparison = {0: 0, 1: 0}
same_ts = 0
same_ts_indices = []
eigval_threshold, deltaG_threshold = 1, 1
diff_eig_nums, diff_deltaG_f_nums, diff_deltaG_r_nums = 0, 0, 0

# Iterate over indices
for index in indices:
    all_present = {0: True, 1: True}
    for ts_type in [0, 1]:
        for calc_type in ["TS", "firc", "rirc"]:
            if index not in master_dict[ts_type][calc_type]:
                all_present[ts_type] = False
        if all_present[ts_type]:
            num_all_type_present[ts_type] += 1

    if all_present == {0: True, 1: True}:
        num_all_present += 1
        good_indices.append(index)

# Perform comparisons
for index in good_indices:
    check0f0r = compare_mols(master_dict[0]["firc"][index]["mol"], master_dict[0]["rirc"][index]["mol"])
    check1f1r = compare_mols(master_dict[1]["firc"][index]["mol"], master_dict[1]["rirc"][index]["mol"])
    check0f1f = compare_mols(master_dict[0]["firc"][index]["mol"], master_dict[1]["firc"][index]["mol"])
    check0r1r = compare_mols(master_dict[0]["rirc"][index]["mol"], master_dict[1]["rirc"][index]["mol"])
    check0f1r = compare_mols(master_dict[0]["firc"][index]["mol"], master_dict[1]["rirc"][index]["mol"])
    check1f0r = compare_mols(master_dict[1]["firc"][index]["mol"], master_dict[0]["rirc"][index]["mol"])
    #print(check0f0r)
    if not check0f0r:
        iso_checks[0] += 1
        print(check0f0r)
    #if not check1f1r:
    #    iso_checks[1] += 1
    #if not check0f0r and not check1f1r:
    #    if check0f1f and check0r1r:
    #        same_ts += 1
    #        iter_comparison[0] += master_dict[0]["TS"][index]["n_iters"]
    #        iter_comparison[1] += master_dict[1]["TS"][index]["n_iters"]
    #        same_ts_indices.append(index)
    #        eigval0 = np.min(master_dict[0]["TS"][index]["hess_eigvalues"])
    #        eigval1 = np.min(master_dict[1]["TS"][index]["hess_eigvalues"])

    #        if abs(eigval0 - eigval1) > eigval_threshold:
    #            diff_eig_nums += 1

    #        gibbs_ts0 = master_dict[0]["TS"][index]["gibbs_free_energy"]
    #        gibbs_f0 = master_dict[0]["firc"][index]["gibbs_free_energy"]
    #        gibbs_r0 = master_dict[0]["rirc"][index]["gibbs_free_energy"]

    #        gibbs_ts1 = master_dict[1]["TS"][index]["gibbs_free_energy"]
    #        gibbs_f1 = master_dict[1]["firc"][index]["gibbs_free_energy"]
    #        gibbs_r1 = master_dict[1]["rirc"][index]["gibbs_free_energy"]
    #        delta_G0_f = gibbs_ts0 - gibbs_f0
    #        delta_G0_r = gibbs_ts0 - gibbs_r0

    #        delta_G1_f = gibbs_ts1 - gibbs_f1
    #        delta_G1_r = gibbs_ts1 - gibbs_r1
    #        if abs(delta_G0_f - delta_G1_f) > deltaG_threshold:
    #            diff_deltaG_f_nums += 1
    #        if abs(delta_G0_r - delta_G1_r) > deltaG_threshold:
    #            diff_deltaG_r_nums += 1

    #elif check0f1r and check1f0r:
    #    same_ts += 1
    #    iter_comparison[0] += master_dict[0]["TS"][index]["n_iters"]
    #    iter_comparison[1] += master_dict[1]["TS"][index]["n_iters"]
    #    same_ts_indices.append(index)
    #    eigval0 = np.min(master_dict[0]["TS"][index]["hess_eigvalues"])
    #    eigval1 = np.min(master_dict[1]["TS"][index]["hess_eigvalues"])
    #    if abs(eigval0 - eigval1) > eigval_threshold:
    #        diff_eig_nums += 1
    #    gibbs_ts0 = master_dict[0]["TS"][index]["gibbs_free_energy"]
    #    gibbs_f0 = master_dict[0]["TS"][index]["gibbs_free_energy"]
    #    gibbs_r0 = master_dict[0]["TS"][index]["gibbs_free_energy"]

    #    gibbs_ts1 = master_dict[1]["TS"][index]["gibbs_free_energy"]
    #    gibbs_f1 = master_dict[1]["TS"][index]["gibbs_free_energy"]
    #    gibbs_r1 = master_dict[1]["TS"][index]["gibbs_free_energy"]
    #    delta_G0_f = gibbs_ts0 - gibbs_f0
    #    delta_G0_r = gibbs_ts0 - gibbs_r0

    #    delta_G1_f = gibbs_ts1 - gibbs_f1
    #    delta_G1_r = gibbs_ts1 - gibbs_r1
    #    if abs(delta_G0_f - delta_G1_f) > deltaG_threshold:
    #        diff_deltaG_f_nums += 1
    #    if abs(delta_G0_r - delta_G1_r) > deltaG_threshold:
    #        diff_deltaG_r_nums += 1

#print('num_all_type_present', num_all_type_present)
#print('num_all_present', num_all_present)
#print('iso_checks', iso_checks)
#print('same_ts', same_ts)
#print('iter_comparison', iter_comparison)
#print('diff_eig_nums', diff_eig_nums)
#print('diff_deltaG_f_nums', diff_deltaG_f_nums)
#print('diff_deltaG_r_nums', diff_deltaG_r_nums)
#
#data = np.zeros((len(same_ts_indices), 2))
#for ii, index in enumerate(same_ts_indices):
#    data[ii, 0] = master_dict[0]["TS"][index]["n_iters"]
#    data[ii, 1] = master_dict[1]["TS"][index]["n_iters"]

