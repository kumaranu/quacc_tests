import numpy as np
import matplotlib.pyplot as plt
import pytest
from graphs_analysis import (
    plot_mismatch_data,
    plot_graph_comps,
    plot_eigval_comps,
    plot_niter_comps
)

"""

# Define a fixture to load the test data
@pytest.fixture
def test_data():
    graph_comps = np.array([[1, 2, 2],
                            [2, 3, 3],
                            [3, 4, 4],
                            [4, 5, 6]])
    data1 = np.array([[1, 5],
                      [2, 6],
                      [3, 7],
                      [4, 8]])
    data2 = np.array([[1, 4],
                      [2, 6],
                      [3, 7],
                      [4, 8]])
    return graph_comps, data1, data2


def test_plot_mismatch_data(test_data):
    graph_comps, data1, data2 = test_data
    threshold_low = 2
    threshold_high = 7
    plt.figure()  # Create a new figure to avoid overlapping plots
    plot_mismatch_data(graph_comps,
                       data1,
                       data2,
                       threshold_low,
                       threshold_high)

    # Check if the plot was created successfully
    assert plt.gcf().get_axes(), "No axes found in the plot"

    # Add more assertions as needed to validate the plot content, labels, legends, etc.


def test_plot_graph_comps(test_data):
    graph_comps, _, _ = test_data
    plt.figure()  # Create a new figure to avoid overlapping plots
    plot_graph_comps(graph_comps)

    # Check if the plot was created successfully
    assert plt.gcf().get_axes(), "No axes found in the plot"

    # Add more assertions as needed to validate the plot content, labels, legends, etc.


def test_plot_eigval_comps(test_data):
    _, data1, data2 = test_data
    threshold_low = 4
    threshold_high = 8
    plt.figure()  # Create a new figure to avoid overlapping plots
    plot_eigval_comps(data1, data2, threshold_low, threshold_high)

    # Check if the plot was created successfully
    assert plt.gcf().get_axes(), "No axes found in the plot"

    # Add more assertions as needed to validate the plot content, labels, legends, etc.


def test_plot_niter_comps(test_data):
    _, data1, data2 = test_data
    threshold_low = 0
    threshold_high = 10
    plt.figure()  # Create a new figure to avoid overlapping plots
    plot_niter_comps(data1, data2, threshold_low, threshold_high)

    # Check if the plot was created successfully
    assert plt.gcf().get_axes(), "No axes found in the plot"

    # Add more assertions as needed to validate the plot content, labels, legends, etc.

# If needed, you can add more test cases for other functions as well

"""
