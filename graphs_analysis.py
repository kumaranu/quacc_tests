import matplotlib.pyplot as plt
import numpy as np


def plot_mismatch_data(graph_comps, data1, data2, threshold_low, threshold_high):
    x_graph = graph_comps[:, 0]
    y1_graph = graph_comps[:, 1]
    y2_graph = graph_comps[:, 2]
    mismatch_indices = np.where(y1_graph != y2_graph)[0]
    print(len(mismatch_indices))
    x_data1 = data1[:, 0]
    y1_data1 = data1[:, 6]
    x_data2 = data2[:, 0]
    y1_data2 = data2[:, 6]

    x_filtered1 = x_data1[mismatch_indices]
    y1_filtered1 = y1_data1[mismatch_indices]
    x_filtered2 = x_data2[mismatch_indices]
    y1_filtered2 = y1_data2[mismatch_indices]

    plt.plot(x_filtered1, y1_filtered1, label='Custom Hessian', color='blue')
    plt.scatter(x_filtered2, y1_filtered2, label='No custom Hessian', color='red')

    plt.xlabel('Index')
    plt.ylabel('7th Column')
    plt.title('Mismatched Data Comparison')

    plt.legend()
    #plt.show()

    plt.savefig('mismatch_graph_eigs.png', dpi=300)
    plt.close()
'''

def plot_same_graph_data(graph_comps, data1, data2, threshold_low, threshold_high):
    x_graph = graph_comps[:, 0]
    y1_graph = graph_comps[:, 1]
    y2_graph = graph_comps[:, 2]
    mismatch_indices = np.where((y1_graph=1) or (y2_graph=1)[0]

    x_data1 = data1[:, 0]
    y1_data1 = data1[:, 6]
    x_data2 = data2[:, 0]
    y1_data2 = data2[:, 6]

    x_filtered1 = x_data1[mismatch_indices]
    y1_filtered1 = y1_data1[mismatch_indices]
    x_filtered2 = x_data2[mismatch_indices]
    y1_filtered2 = y1_data2[mismatch_indices]

    plt.plot(x_filtered1, y1_filtered1, label='Custom Hessian', color='blue')
    plt.scatter(x_filtered2, y1_filtered2, label='No custom Hessian', color='red')

    plt.xlabel('Index')
    plt.ylabel('7th Column')
    plt.title('Mismatched Data Comparison')

    plt.legend()
    #plt.show()

    plt.savefig('same_graph_graph_eigs.png', dpi=300)
    plt.close()
'''

def plot_graph_comps(graph_comps):
    x  = graph_comps[:, 0]
    y1 = graph_comps[:, 1]
    y2 = graph_comps[:, 2]

    plt.scatter(x, y1, label='Custom Hessian', color='blue', marker='x', s=20)
    plt.scatter(x, y2, label='No custom Hessian', color='green', marker='o', s=4)

    mismatch_indices = np.where(y1 != y2)[0]
    for index in mismatch_indices:
        plt.axvline(x=x[index], color='red', linestyle='--')

    plt.xlabel('Test reaction index')
    plt.ylabel('Molecular graph isomorphism')
    plt.title('Comparison between molecular graphs for the IRC end-points')

    plt.legend()

    y_tick_labels = ['True', 'False']
    plt.yticks([True, False], y_tick_labels)

    plt.savefig('graph_comparisons.png', dpi=300)
    plt.close()


def plot_eigval_comps(data1, data2, threshold_low, threshold_high):
    x  = data1[:, 0]
    y1 = data1[:, 6]
    y2 = data2[:, 6]

    mask1 = np.logical_and(y1 >= threshold_low, y1 <= threshold_high)
    mask2 = np.logical_and(y2 >= threshold_low, y2 <= threshold_high)
    x_filtered1 = x[mask1]
    y1_filtered = y1[mask1]
    x_filtered2 = x[mask2]
    y2_filtered = y2[mask2]

    plt.plot(x_filtered1, y1_filtered, label='Custom Hessian', color='blue')
    plt.scatter(x_filtered2, y2_filtered, label='No custom Hessian', color='red')

    plt.xlabel('Test reaction index')
    plt.ylabel('Negative eigenvalues (eV) for Hessian')

    plt.legend()

    plt.savefig('eig_comparisons.png', dpi=300)
    plt.close()


def plot_niter_comps(data1, data2, threshold_low, threshold_high):
    x  = data1[:, 0]
    y1 = data1[:, 1]
    y2 = data2[:, 1]

    mask1 = np.logical_and(y1 >= threshold_low, y1 <= threshold_high)
    mask2 = np.logical_and(y2 >= threshold_low, y2 <= threshold_high)
    x_filtered1 = x[mask1]
    y1_filtered = y1[mask1]
    x_filtered2 = x[mask2]
    y2_filtered = y2[mask2]

    plt.plot(x_filtered1, y1_filtered, label='Custom Hessian', color='blue')
    plt.plot(x_filtered2, y2_filtered, label='No custom Hessian', color='red')

    plt.xlabel('Test reaction index')
    plt.ylabel('Number of iterations in transition state calculation')

    plt.legend()

    plt.savefig('n_iter_comparisons.png', dpi=300)
    plt.close()

if __name__ == '__main__':
    graph_comps = np.loadtxt('graph_comps.txt')
    all_analysis_data0TS = np.loadtxt('all_analysis_data0-TS.txt')
    all_analysis_data1TS = np.loadtxt('all_analysis_data1-TS.txt')

    plot_graph_comps(graph_comps)
    plot_eigval_comps(all_analysis_data0TS, all_analysis_data1TS, -50, -3.6)
    plot_niter_comps(all_analysis_data0TS, all_analysis_data1TS, 0, 99)

    plot_mismatch_data(graph_comps, all_analysis_data0TS, all_analysis_data1TS, -50, -3.6)
    #plot_same_graph_data(graph_comps, all_analysis_data0TS, all_analysis_data1TS, -50, -3.6)


def plot_mismatch_data(graph_comps, data1, data2, threshold_low, threshold_high):
    x  = graph_comps[:, 0]
    y1 = graph_comps[:, 1]
    y2 = graph_comps[:, 2]
    mismatch_indices = np.where(y1 != y2)[0]

    x  = data1[:, 0]
    y1 = data1[:, 6]
    y2 = data2[:, 6]

    mask1 = np.logical_and(y1 >= threshold_low, y1 <= threshold_high)
    mask2 = np.logical_and(y2 >= threshold_low, y2 <= threshold_high)
    x_filtered1 = x[mask1]
    y1_filtered = y1[mask1]
    x_filtered2 = x[mask2]
    y2_filtered = y2[mask2]

    plt.plot(x_filtered1, y1_filtered, label='Custom Hessian', color='blue')
    plt.scatter(x_filtered2, y2_filtered, label='No custom Hessian', color='red')


