# Medina (2007) StOT model: basic

This script computes a model of the grammaticality of implicit objects based on Medina (2007)'s variant of Stochastic Optimality Theory (StOT).

Medina's analysis accounts for the gradient grammaticality of an implicit object across verbs, relying on StOT's floating constraints system. The input to the model has to contain the verb, its telicity, the perfectivity of the sentence, and the SPS of the verb. Medina's variant of StOT defines the re-ranking probabilities as a function of SPS, instead of assigning all constraints the same Gaussian distribution.

## Getting Started
The script is standalone and runs on Python 3. It has been tested on Ubuntu 20.10.

### Prerequisites
You may need to install the following packages to make the script work:

    argparse>=1.1
    pandas>=0.25.3
    numpy>=0.25.3
    matplotlib>=3.1.2
    scipy>=1.3.3
    seaborn>=0.10.0
    statsmodels>=0.11.1
    statannot>=0.2.3

To install these packages in Python 3, make sure you have installed pip3 and run:

    pip3 install <package>

## Running the script
Be sure you have the script and the input folder within the same folder before starting.

### Characteristics of the input data

#### SPS files
Provide header-less, space-separated SPS files having verbs in the first column and SPS scores in the second column.

#### Judgments file
Provide headed, tab-separated judgment files having the following columns (handily, but not necessarily, in this order):

* "verb": verb names
* "sentence": sentence type based on experimental setting, may take the following values
    * `target`: verbs of interest, no object
    * "control": verbs of interest, overt object
    * "filler_no": intransitive verbs, no object
    * "filler_dobj": intransitive verbs, overt object
* "telicity": verb telicity, may be either "telic" or "atelic"
* "perfectivity": sentence perfectivity, may be either "perf" or "imperf"
* "s1, s2, s3... sN": a column for each participant to the experiment, numbered progressively, containing their raw Likert-scale judgments. This script has been tested on 7-point Likert judgments, but it will work with any Likert scale you choose.

Moreover, you may include two columns (either, both, or neither) that are going to be used in extended versions of this model, namely:
* "iterativity": sentence iterativity, may be either "iter" or "noiter"
* "mannspec": verb manner specification, may be either "spec" or "nospec"
They are useless for this script's purposes, but if you run the experiment including them in your design, you don't have to reshape your input to run the basic model.

### Parameters
To perform a quick test run on the mock input data included in this repository, just run this under the main directory:

    python3 optimizeMedinaBasic.py

You may pass several optional parameters to the script:

    --sps, -s:        folder containing SPS files
    --judgments, -j:  file containing raw acceptability judgments
    --output, -o:     output folder
    
To access the list of parameters in your terminal, run:

    python3 optimizeMedinaBasic.py -h
    
For instance, to run the script on the mock input data included in this repository by specifying each parameter, you would run:

    python3 optimizeMedinaBasic.py -s input/sps/ -j input/judgments/mock_judgments.csv -o output/
    
To run it on the extended mock input data (getting the exact same output!), you would instead run:

    python3 optimizeMedinaBasic.py -s input/sps/ -j input/judgments/mock_judgments_ext.csv -o output/

### Output
The script prints out in the output folder everything you need as a linguist to describe your input data and the output of the model you built.

#### Terminal output
While the script is running, it prints human-friendly comments in stdout so you can follow its progress.

#### Preliminary data exploration
First of all, you want to make sure the judgments you collected have a nice shape. The script plots a series of boxplots to do so, one for each verb, for each *sentence* type you provided in the input (results for filler sentences are collapsed). You can find these boxplots in:

    output/preliminary/

#### Output for each SPS file in input/sps/
testo

## License
This project is licensed under the MIT License.

## References
* Medina, Tamara Nicol (2007). Learning which verbs allow object omission: verb semantic selectivity and the implicit object construction (Doctoral dissertation, Johns Hopkins University).
* Kim, Najoung; Rawlins, Kyle; Smolensky, Paul (2019). "The complement-adjunct distinction as gradient blends: the case of English prepositional phrases", [lingbuzz/004723](https://ling.auf.net/lingbuzz/004723)

## Acknowledgments
* testo
* testo
