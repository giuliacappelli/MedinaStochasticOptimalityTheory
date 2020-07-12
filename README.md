# Medina (2007) StOT model: basic

This script computes a model of the grammaticality of implicit objects/arguments based on Medina (2007)'s variant of Stochastic Optimality Theory.

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
testo

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

### Output
testo

## License
This project is licensed under the MIT License.

## References
* testo
* testo

## Acknowledgments
* testo
* testo
