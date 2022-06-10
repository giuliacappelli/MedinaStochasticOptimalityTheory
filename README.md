# MedStOT model: modeling the grammaticality of implicit objects based on Medina (2007)'s variant of Stochastic Optimality Theory

This project computes a model of the grammaticality of implicit objects based on Medina (2007)'s variant of Stochastic Optimality Theory (StOT).

Medina's analysis accounts for the gradient grammaticality of an implicit object across verbs, relying on StOT's floating constraints system. The input to the model has to contain the verb, its telicity and manner specification features, the perfectivity and iterativity features of the sentence, and the SPS of the verb. Medina's variant of StOT defines the re-ranking probabilities as a function of SPS, instead of assigning all constraints the same Gaussian distribution.

## Getting Started
The scripts should run on Python 3.0+ and do not have to be installed. They run fine on Python 3.8.2 in Ubuntu 20.10.

### Prerequisites
You need the following packages to make the scripts work:

    argparse>=1.1
    pandas>=0.25.3
    numpy>=1.17.4
    matplotlib>=3.1.2
    scipy>=1.3.3
    seaborn>=0.10.0
    statsmodels>=0.11.1
    statannot>=0.2.3
    tikzplotlib>=0.10.1

To install these packages in Python 3, make sure you have installed pip3 and run:

    pip3 install <package>
    
### Scripts in this project
This project contains three scripts, one for each model of the grammaticality of implicit objects I am going to build:
* `optimizeMedinaBasic.py` reproduces Medina's original model, whose constraints are \*IntArg, FaithArg, PerfCoda, and TelicEnd
* `optimizeMedinaExtended1.py` computes the original model with an additional constraint, NonIterArg (Non-Iterative Argument)
* `optimizeMedinaExtended2.py` computes the Extended1 model with an additional constraint, MannSpecArg (Manner-Specified Argument)

## Running the scripts
Make sure you have the scripts and the input folder within the same folder before starting. You may run these scripts on three different kinds of input data:
* the mock input data included in this repository for your convenience
* the actual input data I used in my dissertation, freely available and documented [here](https://github.com/giuliacappelli/dissertationData)
* mock data you can generate (and fiddle with) [with my script](https://github.com/giuliacappelli/generateMockLikertGrammaticalityJudgments) to suit your testing needs before running an experiment of your own

### Characteristics of the input data

#### SPS files
In `input/sps/`, provide header-less, space-separated SPS files having verbs in the first column and SPS scores in the second column. For instance,

| | |
|-|-|
| eat | 5.27
| approve | 3.50
| like | 0.71

#### Zipf scores files
In `input/zipf/`, provide header-less, space-separated SPS files having verbs in the first column and Zipf scores (base 10 logarithm of the
frequency-per-billion-words) in the second column. For instance,

| | |
|-|-|
| eat | 5.27
| approve | 3.50
| like | 2.71

#### Judgments file
In `input/judgments/`, provide headed, tab-separated judgment files having the following columns (handily, but not necessarily, in this order):

* `verb`: verb names
* `sentence`: sentence type based on experimental setting, may take the following values
    * `target`: verbs of interest, no object
    * `control`: verbs of interest, overt object
    * `filler_no`: intransitive verbs, no object
    * `filler_dobj`: intransitive verbs, overt object
* `telicity`: verb telicity, may be either "telic" or "atelic"
* `perfectivity`: sentence perfectivity, may be either "perf" or "imperf"
* `s1, s2, s3... sN`: a column for each participant to the experiment, numbered progressively, containing their raw Likert-scale judgments. This script has been tested on 7-point Likert judgments, but it will work with any Likert scale you choose.
* `iterativity`: sentence iterativity, may be either "iter" or "noiter" (OPTIONAL to run `optimizeMedinaBasic.py`)
* `mannspec`: verb manner specification, may be either "spec" or "nospec" (OPTIONAL to run `optimizeMedinaExtended1.py`)

The scripts take care of excluding any optional column among these from the analysis, so that you don't need to do that manually.

For instance, your input data will be shaped like this:

verb | sentence | telicity | perfectivity | iterativity | mannspec | s1 | s2 | s3
|-|-|-|-|-|-|-|-|-|
eat | target | atelic | perf | iter | nospec | 4 | 5 | 4
eat | target | atelic | imperf | iter | nospec | 6 | 6 | 7
kill | target | telic | perf | iter | nospec | 2 | 3 | 2
kill | target | telic | imperf | iter | nospec | 5 | 4 | 5

### Parameters
To perform a quick test run on the mock input data included in this repository, just run this under the main directory:

    python3 optimizeMedina[Basic|Extended1|Extended2].py

You may pass several optional parameters to the scripts:

    --sps, -s:        folder containing SPS files (defaults to input/sps/)
    --judgments, -j:  file containing raw acceptability judgments (defaults to input/judgments/mock_judgments_ext.csv)
    --zipf, -z:       file containing Zipf scores (defaults to input/judgments/zipf.csv)
    --output, -o:     output folder (defaults to output/)
    
To access the list of parameters in your terminal, run:

    python3 optimizeMedina[Basic|Extended1|Extended2].py -h
    
For instance, to run the script on the mock input data included in this repository by specifying each parameter, you would run:

    python3 optimizeMedina[Basic|Extended1|Extended2].py -s input/sps/ -j input/judgments/mock_judgments_ext.csv -z input/judgments/zipf.csv -o output/

### Preprocessing
The script will take care of preprocessing your input data (these innovations are not in Medina 2007). 

In particular, it will compute min-max normalized (or z-scores of) SPS values for each SPS input file to make results comparable across models. You can also choose to use raw SPS data by commenting out a paragraph in the script, at your own risk.

Most importantly, I preprocessed raw judgment data following Kim et al. (2019). I computed the within-subject z-scores for the judgments, then averaged these scores to obtain the mean judgment for each sentence in the stimuli list, then normalized the mean judgments between 0 and 1.

### Output
The script prints out in the output folder everything you need as a linguist to describe your input data and the output of the model you built. As a treat, the scripts save each plot both in png and in tikz (LaTeX) format, to suit your every publication need.

#### Terminal output
While the script is running, it prints human-friendly comments in stdout so you can follow its progress.

#### Preliminary data exploration
First of all, you want to make sure the judgments you collected have a nice shape. The script plots a series of boxplots to do so, one for each verb, for each *sentence* type you provided in the input (results for filler sentences are collapsed). You can find these boxplots in `output/preliminary/plot_boxplot_[sentence_type].png`.

Before computing any model with your data, you also want to check your hypotheses first. You can see whether your aspectual types determine any difference in the median judgments in `output/preliminary/plot_preliminary_boxplot_[aspectual_type].png`, and you can visualize the effect of SPS on judgments in  `output/preliminary/plot_preliminary_scatterplot_[sps_filename].png`.

Of course, you are also interested in the combined effect of your predictors on the gradient grammaticality judgments you collected. Medina (2007) accomplished this with a multiple regression, and I up the game with a linear mixed-effects model (LMEM). You can find the result table in `output/preliminary/lmem_[sps_filename].txt`. The R-style formula for the LMEM is:

    judgment ~ sps + telicity + perfectivity + iterativity + mannspec + (1|verb) + (1|subject)
    
and input judgments are *raw* ones instead of within-subject z-scores, since the LMEM will take care of that.
    
Feel more confident reading your LMEMs' results in R? Need some coefficient that Python does not yield? Fear not: `output/preliminary/dataframe_input_lmem_[sps_filename].csv` contains the dataframe we used as input to the LMEM, so you can just open RStudio, plug that in, and run your analysis. I tested this in R 4.0.2 with lme4:lmer() and got the same results as in my Python script.

#### Output of the model proper
Now we're all set to compute Medina's model. For each SPS input file, you find the model's results in `output/[sps_filename]/`. This folder(s) contain several items:
* `constantsMedina.txt`: list of deltas and gammas computed by fitting Medina's model on input data (puzzled? read Medina 2007 to get it, it's crucial!)
* `errors.txt`: individual squared errors (to quantify the difference between actual and predicted judgments) and adjusted R squared values for the whole model
* `pearson.txt`: Pearson's r and p-value between actual and predicted judgments, a line for each aspectual type in the input (and the overall value for the model)
* `plot_prob_[constraint].png`: plot to visualize the probability of \*INT ARG reranking with each of the other constraints, based on each pair of deltas and gammas
* `plot_prob_aspectualtypes.png`: plot to visualize the probability of an implicit object output for each aspectual type, based on the model results

That's it, folks! Hope you liked running these scripts, I sure loved writing them :smiling_face_with_three_hearts:

## License
This project is licensed under the MIT License.

## References
* Medina, Tamara Nicol (2007). Learning which verbs allow object omission: verb semantic selectivity and the implicit object construction (PhD dissertation, Johns Hopkins University).
* Kim, Najoung; Rawlins, Kyle; Smolensky, Paul (2019). "The complement-adjunct distinction as gradient blends: the case of English prepositional phrases", [lingbuzz/004723](https://ling.auf.net/lingbuzz/004723)

## Acknowledgments
Many thanks to 
* @ellepannitto, my Python fairy
* @najoungkim, for sharing references that ultimately led to this project
* the Stack Overflow community, for the many code snippets that saved me from frustration

## Known issues
The scripts work fine, but they are too long, non adaptable to different inputs (e.g. inputs with more constraints, or differently named cells), and resource-heavy. Most importantly, they are TOO MANY: a single, well-written script would suffice. I did not work on these issues since I only need this project to run on small-ish judgment dataframes in three different flavors, but this is definitely something that needs to be done for elegance's sake.
