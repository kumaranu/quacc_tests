# Quacc Tests

This repository contains wrapper scripts for running transition state and IRC (Intrinsic Reaction Coordinate) calculations using Sella and IRC ASE optimizers for the Sella package.

## Overview

- `nn_sella_quacc.py`: Wrapper script for NewtonNet-based optimizations.
- `dft_sella_quacc.py`: Wrapper script for DFT (Density Functional Theory) method, specifically using the wb97x/6-31G* level of theory.

## Input Configuration

The `inputs` directory includes two configuration files:
- `config43.toml`: Input configuration for NewtonNet-based optimizations.
- `config44.toml`: Input configuration for DFT calculations.

## Running Locally

While these scripts may require significant setup for execution on a supercomputer and storing results in a database, the `run_locally` function within these scripts enables running without a major time investment for new users.

## Usage

To run the scripts locally, simply execute the desired script, e.g.,

```bash
python nn_sella_quacc.py

## Note

For detailed information on setup and configuration, please refer to the following:

1. **Sella Package:**
   - [GitHub Repository](https://github.com/zadorlab/sella)
   - [Documentation](https://github.com/zadorlab/sella/wiki)

2. **NewtonNet:**
   - [GitHub Repository](https://github.com/THGLab/NewtonNet)

3. **QuAcc Recipes for NewtonNet and QChem:**
   - [GitHub Repository](https://github.com/Quantum-Accelerators/quacc/blob/main/src/quacc/recipes/newtonnet/ts.py)
   - [Documentation](https://quantum-accelerators.github.io/quacc/reference/quacc/recipes/newtonnet/ts.html)

4. **Corresponding Paper Authors:**
   - Feel free to reach out to them (including me) for assistance.
