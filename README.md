# axion-miniclusters

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4006128.svg)](https://doi.org/10.5281/zenodo.4006128) [![arXiv](https://img.shields.io/badge/arXiv-2011.05377-B31B1B)](http://arxiv.org/abs/2011.05377) [![arXiv](https://img.shields.io/badge/arXiv-2011.05378-B31B1B)](http://arxiv.org/abs/2011.05378) ![GitHub](https://img.shields.io/badge/miniclusters-perturbed-green) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*Code and results for the disruption of axion miniclusters (AMCs) in the Milky Way, as well as radio signals from encounters between AMCs and neutrons stars.*

The key parts of the computation are:
* Calculation of the perturbations to individual AMCs due to stellar encounters ([`code/Distribution_PL.ipynb`](code/Distribution_PL.ipynb) and [`code/Distribution_NFW.ipynb`](code/Distribution_NFW.ipynb)) 
* Monte Carlo simulations of the disruption of individual AMCs orbiting in the Milky Way ([`code/MC_script_ecc.py`](code/MC_script_ecc.py))  
* Processing and calculation of the distributions of AMC properties from the Monte Carlo results ([`code/prepare_distributions.py`](code/prepare_distributions.py))  
* Sampling of properties of individual encounters between AMCs and neutrons stars in the Milky Way ([`code/simulate_signal.py`](code/simulate_signal.py))

Scripts for generating plots from the results are in [`code/plotting/`](code/plotting). The first thing to do is to edit [`code/dirs.py`](code/dirs.py) so that the directory variables point to the right place.

The raw Monte Carlo results are archived online at https://doi.org/10.6084/m9.figshare.13224386.v1. Edit the file [`code/dirs.py`](code/dirs.py) to specify the directory where these Monte Carlo results are located (though the raw files are only needed if you want to re-calculate the AMC distributions).

Full samples of radio signal events due to AMC-NS encounters are archived online at https://doi.org/10.6084/m9.figshare.13204856.v1. These should be placed in the [`data/`](data/) folder. The full samples contain 10^7 events each; without these the plotting scripts will use instead the 'short' sample files provided, each of which is 10^5 events.

### Re-interpreting the results

If you would like to re-compute the AMC distributions assuming a different axion mass or initial mass function, the mass function can be specified in [`code/mass_function.py`](code/mass_function.py). For a power-law mass function, you can simply edit the command
```python
AMC_MF = mass_function.PowerLawMassFunction(m_a, gamma)
```
in the scripts [`code/prepare_distributions.py`](code/prepare_distributions.py) and [`code/simulate_signal.py`](code/simulate_signal.py). The new AMC distributions and signal samples can then be computed with 
```bash
./GetDistributions.sh
```
and 
```bash
./GetSignals.sh
```
Note that these commands will replace the distribution and signal sample files which are already in the repository. You can specify the `IDstr` variable hard-coded in [`code/prepare_distributions.py`](code/prepare_distributions.py) and [`code/simulate_signal.py`](code/simulate_signal.py) to add a file suffix so that the old files are not overwritten.

**Setting different parameters:** Inevitably, you'll want to specify different parameters to the simulations, as well as differentiating between different simulations. The file [`code/params.py`](code/params.py) allows you to do this relatively straight-forwardly. At the moment, it allows you to specify:
- The axion mass `m_a`  
- The minimum density enhancement `k` to consider inside axion miniclusters (that is, encounters for which the AMC internal density does not exceed the smooth Galactic background density by at least a factor `k` will be ignore)  
- An identification string, which will be used to save output files: `IDstr`.

### Generating quick results with delta-function mass functions

As of July 2021, new scripts have been added for preparing distributions and simulating signals for *delta-function mass functions of AMCs* - [`code/prepare_distributions_delta.py`](code/prepare_distributions_delta.py) and [`code/simulate_signal_delta.py`](code/simulate_signal_delta.py). These scripts accept the flag `-mass_choice` with `-mass_choice c` fixing the AMC mass to be equal to the characteristic AMC mass `M_0` and `-mass_choice a` fixing the AMC mass to be equal to the mean AMC mass, given a power-law mass function. The files which are output by these scripts will have filenames appended with `_delta_a` or `_delta_c`, along with whatever value of `IDstr` is specified in the `params.py` file. 

To run these scripts, you'll need to download the raw Monte Carlo data files mentioned above. However, you shouldn't need to re-run Monte Carlos, as the characteristic and mean AMC masses are almost always well within the range of masses which were simulated. 

As a rough example of running the full pipeline in this case, you could run:
```bash
python3 prepare_distributions_delta.py -profile PL -mass_choice a -AScut -max_rows 10000
python3 simulate_signal_delta.py -Ne 1e5 -profile PL -mass_choice a -AScut
```
This should save a list of encounters in the `data/` folder (in two different formats, depending on how you want to use them). You can also then run [`code/plotting/CalcEncounterRate.py`](code/plotting/CalcEncounterRate.py) (passing the appropriate `-IDstr` flag) to calculate the total encounter rate.


### Citation

If you use the code or the associated data, please cite this repository and its DOI: [10.5281/zenodo.4006128](https://doi.org/10.5281/zenodo.4006128).

Please also cite the two associated papers:
> B. J. Kavanagh, T. D. P. Edwards, L. Visinelli & C. Weniger (2020), "Stellar Disruption of Axion Miniclusters in the Milky Way".

>T. D. P. Edwards, B. J. Kavanagh, L. Visinelli & C. Weniger (2020), "Transient Radio Signatures from Neutron Star Encounters with QCD Axion Miniclusters".

### Authors & Contact

Written and maintained by Bradley J. Kavanagh, Thomas D. P. Edwards and Luca Visinelli.

This repository contains code which is being actively used for research, so in places it may not be 100% clear. If you have any questions whatsoever, or if the code behaves in an unexpected way, please do not hesitate to contact the authors (e.g. at bradkav@gmail.com).

### License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
