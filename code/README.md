# Code

A more detailed guide to using the code, and especially adapting the code for new mass functions and AMC density profiles.

**1. Specify the directories**

In `code/dirs.py`, change the `data_dir` variable to point to the `data` directory in the `axion-minicluster` repo. Change the `montecarlo_dir` to point to wherever you want the Monte Carlo samples to be stored (it's sometimes useful to keep these in a different place; they tend to take up a lot of memory).

**2. Calculate the AMC response functions**

Specify the internal AMC density profile in the jupyter notebook `Distribution_generic.ipynb`. Then run the whole notebook. This will calculate some internal properties of the AMC and determine the response of the AMC to perturbations. These are output to a few files (`data/AMC_parameters_XXX.txt` and `data/Perturbations_XXX.txt`, where XXX is the label for the profile) At the moment, the code only supports a single, universal shape of density profile for the AMCs. 

**3. Specify the AMC mass function**

Define a new mass function class inside `mass_function.py`. Take a look at the (completely made up) `ExampleMassFunction` class. Your mass function should inherent from the `GenericMassFunction` class, and it should define a few class variables, including `mmin`, `mmax` (range of AMC masses), `rhomin`, `rhomax` (range of AMC mean internal densities), `mavg` (mean AMC mass). You should also define a few functions, which are the PDF for the mean internal AMC density (`dPdrho`), the AMC mass (`dPdlogM`) and the joint PDF (`dPdlogMdrho`).

Functions in the `GenericMassFunction` class are then available to sample the AMC mass and density. 

Once you've defined your mass function, edit a couple of scripts to point to this mass function. In particular, `MC_script_ecc.py` and `prepare_distributions.py` (look for the comment `#ACTION: Specify mass function`).


**4. Run the code**

Let's assume for now that you've defined a new internal density profile labelled `NFW_test`. 

The core of the Monte Carlo code is in `MC_script_ecc.py`, which generates a sample of AMCs are a given semi-major axis and then simulates their disruption. Run
```bash
python RunFullMonteCarlo.py
```
to run the MonteCarlo over a grid of semi-major axis values. Take a look at `RunFullMonteCarlo.py` for some options which can be specified (such as density profile, and number of AMCs to sample).

You can do some post-processing of the Monte Carlo samples with
```bash
python prepare_distributions.py -profile NFW_test
python prepare_distributions.py -profile NFW_test -unperturbed 1
```
This calculates the AMC distribution as a function of galactocentric radius, and estimates distribution of AMC mass and radius at different positions.

Finally, you can generate a couple of plots (of survival probability and AMC distributions) using:
```bash
cd plotting
python PlotSurvivalProbability_generic.py -profile NFW_test
python PlotDistributions.py -profile NFW_test -R 8.633
```

#### Caveats

- Axion star cut  
- Sampling of the AMC mass  
- Cut off in the AMC density in calculating the encounter rate
- More of the pipeline?

#### Units

In general, masses are in `Msun`, distances in `pc` and occasionally, just to confuse things, speeds are in `km/s`.


