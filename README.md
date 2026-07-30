# synthBOLD

[![Python](https://img.shields.io/badge/Python-3.13-blue)](https://github.com/haenelt/synthBOLD)
[![License](https://img.shields.io/badge/license-%20%20GNU%20GPLv3%20-green?style=plastic)](https://www.gnu.org/licenses/gpl-3.0)
[![Test and formatting](https://github.com/haenelt/synthBOLD/actions/workflows/tests.yml/badge.svg)](https://github.com/haenelt/synthBOLD/actions/workflows/tests.yml)
[![Docs](https://github.com/haenelt/synthBOLD/actions/workflows/docs.yml/badge.svg)](https://github.com/haenelt/synthBOLD/actions/workflows/docs.yml)
[![codecov](https://codecov.io/gh/haenelt/synthBOLD/graph/badge.svg?token=Z3YGVFyUzv)](https://codecov.io/gh/haenelt/synthBOLD)

![synth batch example](https://raw.githubusercontent.com/haenelt/synthBOLD/main/docs/source/_static/synth_batch.svg)


_synthbold_ is a Python package for synthesizing BOLD fMRI data with controllable macrovascular contributions and corresponding ground-truth labels. The generated datasets can be used to develop, validate, and benchmark fMRI analysis methods without requiring labeled in vivo data.

In particular, it was developed to facilitate methods for identifying and mitigating macrovascular contributions in BOLD fMRI, by providing simulated data with known ground truth for evaluating such methods.

Each BOLD fMRI sample is built from a biophysical forward model of the BOLD signal: randomly placed vascular geometries (vessels and other perturbing structures) are assigned magnetic susceptibility values that generate a surrounding field perturbation (ΔBz), and the complex MRI signal is computed under the static dephasing regime [1], i.e., the intra-voxel signal decay caused by static, susceptibility-induced field inhomogeneities around vessels, following the classic vessel-based BOLD simulation framework of [2].

All generative parameters (vessel geometry, size, orientation, susceptibility, decay rates, echo time, magnetic field strength, etc.) are generated with weak priors and high variance sampling rather than tuned to match any specific real dataset. This domain-randomization strategy (cf. [3, 4]) aims to generate a much broader data distribution than any finite set of real fMRI data acquisitions could provide, so that models trained on synthetic data alone generalize to real fMRI data at test time.

## Installation
_synthbold_ can be installed from [pypi](https://pypi.org/) via

```
pip install synthbold
```

## Usage
To generate a batch of synthetic BOLD data with macrovascular contributions, the `synthbold` command can be called from the command line as follows:

```
synthbold --output <output_dir> --n-sample <n_samples> --batch-size <batch_size> --config <config.yaml>
```

See `synthbold --help` for the full list of options.

## Python API
`synthbold` can also be used as a library. `SynthPipeline` is the main entry point. Calling it returns a batch of synthetic BOLD data with macrovascular contributions: magnitude/phase images together with their ground-truth tissue and vessel maps.

```python
from synthbold.config import Config
from synthbold.pipeline import SynthPipeline

config = Config()
pipeline = SynthPipeline(dirname="output", n_samples=100, config=config)
sample, _, params = pipeline(batch_size=4)
```

## Examples
Example code can be found as Jupyter notebooks in the [`notebooks`](https://github.com/haenelt/synthBOLD/tree/main/notebooks) folder.

## Contact
If you have questions, problems or suggestions regarding the `synthbold` package, please feel free to contact [me](mailto:daniel.haenelt@gmail.com).

## References
1. Yablonskiy, D. A., & Haacke, E. M. (1994). Theory of NMR signal behavior in magnetically inhomogeneous tissues: the static dephasing regime. *Magnetic Resonance in Medicine*, 32, 749–763.
2. Boxerman, J. L., et al. (1995). MR contrast due to intravascular magnetic susceptibility perturbations. *Magnetic Resonance in Medicine*, 34, 555–566.
3. Tobin, J., et al. (2017). Domain randomization for transferring deep neural networks from simulation to the real world. *IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*.
4. Billot, B., et al. (2023). SynthSeg: Segmentation of brain MRI scans of any contrast and resolution without retraining. *Medical Image Analysis*, 86, 102789.
