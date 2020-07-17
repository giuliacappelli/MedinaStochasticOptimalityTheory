import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit # equivalent but simpler than least_squares, no need to define the residual(error) function
from scipy.stats import zscore, pearsonr, linregress
import os
import seaborn as sns
import statsmodels.api as sm # to create linear mixed-effects models and linear regression models
from statannot import add_stat_annotation # to statistically annotate plots

my_parser = argparse.ArgumentParser()

def dir_path(path): # defines a correct folder path to use below as the SPS command line argument
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")
                              
my_parser.add_argument('--sps',
                       '-s',
                       action='store',
                       type=dir_path,
                       default='input/sps/',
                       help='folder containing space-separated SPS files')

my_parser.add_argument('--judgments',
                       '-j',
                       action='store',
                       default='input/judgments/mock_judgments_ext.csv',
                       help='file containing tab-separated raw judgments') # Medina's constraints + iterativity + mannspec
                       
my_parser.add_argument('--output',
                       '-o',
                       action='store',
                       default='output/',
                       help='output folder')                       

args = my_parser.parse_args()

print("fasten your seatbelts! \r\n")

# create output folders if they do not exist yet
if not os.path.exists(args.output): 
    os.makedirs(args.output)

df_full = pd.read_csv(args.judgments, sep='\t') # file has headers!

# ~ # checkpoint! your input dataframe may contain more columns than necessary for the model created by this script
# ~ # if so, the script will aggregate your input by averaging judgments where necessary
# ~ # if not, your input will stay the same

# ~ if 'mannspec' in df_full.columns:
	# ~ df_full = df_full.drop(columns=['mannspec'])
# ~ else:
	# ~ df_full = df_full

numeric_cols = df_full.select_dtypes('int64').columns
df_full_zscores = df_full[numeric_cols].apply(zscore) # computes within-subject z-scores
print("preprocessing input data... \r\n")

# create temporary long dataframe with z-scores to plot preliminary density plots
df_zscores = df_full_zscores.join(df_full[df_full.columns.difference(df_full_zscores.columns)])
df_long = df_zscores.melt(id_vars=['perfectivity','sentence','telicity','iterativity','mannspec','verb'], var_name="subject", value_name="judgment") # long df (z-scores)
df_long_raw = df_full.melt(id_vars=['perfectivity','sentence','telicity','iterativity','mannspec','verb'], var_name="subject", value_name="judgment") # long df (raw judgs)

# create boxplots to describe input data (based on aspectual type and experimental design)
if not os.path.exists(args.output+'preliminary/'):    
	os.makedirs(args.output+'preliminary/')
outdir = args.output+'preliminary/'

for typ in df_long.sentence.unique():		# content of "sentence" column in df_long
	df_typ = df_long[df_long.values == typ]
	
	if "filler" in typ:
		
		plt.figure()
		print("drawing awesome boxplots for", typ, "sentences...")
		sns.boxplot(x="judgment", y="verb", color = 'lightgray', data=df_typ, orient="h", dodge=False)
		# ~ plt.xticks([np.amin(df_long["judgment"].to_numpy()), -1.0, -0.5, 0, 0.5, np.amax(df_long["judgment"].to_numpy())])	# manual
		plt.xticks(np.arange(np.amin(df_long["judgment"].to_numpy()), np.amax(df_long["judgment"].to_numpy())+0.5, 0.5))	# range
		plt.savefig(outdir+'plot_boxplot_'+str(typ)+'.png')
		plt.close()
		
	else:
	
		for prf in df_long.perfectivity.unique():	# content of "perfectivity" column in df_long
			df_typprf = df_typ[df_typ.values == prf]
	
			plt.figure()
			print("drawing neat boxplots for", typ, prf, "sentences...")
			sns.boxplot(x="judgment", y="verb", color = 'lightgray', data=df_typprf, orient="h", dodge=False)
			# ~ plt.xticks([np.amin(df_long["judgment"].to_numpy()), -1.0, -0.5, 0, 0.5, np.amax(df_long["judgment"].to_numpy())])	# manual
			plt.xticks(np.arange(np.amin(df_long["judgment"].to_numpy()), np.amax(df_long["judgment"].to_numpy())+0.5, 0.5))	# range
			plt.savefig(outdir+'plot_boxplot_'+str(typ)+str(prf)+'.png')
			plt.close()
	
		for iterat in df_long.iterativity.unique():	# content of "iterativity" column in df_long
			df_typiterat = df_typ[df_typ.values == iterat]
	
			plt.figure()
			print("drawing neat boxplots for", typ, iterat, "sentences...")
			sns.boxplot(x="judgment", y="verb", color = 'lightgray', data=df_typiterat, orient="h", dodge=False)
			# ~ plt.xticks([np.amin(df_long["judgment"].to_numpy()), -1.0, -0.5, 0, 0.5, np.amax(df_long["judgment"].to_numpy())])	# manual
			plt.xticks(np.arange(np.amin(df_long["judgment"].to_numpy()), np.amax(df_long["judgment"].to_numpy())+0.5, 0.5))	# range
			plt.savefig(outdir+'plot_boxplot_'+str(typ)+str(iterat)+'.png')
			plt.close()
	
		for mannsp in df_long.mannspec.unique():	# content of "mannspec" column in df_long
			df_typmannsp = df_typ[df_typ.values == mannsp]
	
			plt.figure()
			print("drawing neat boxplots for", typ, mannsp, "sentences...")
			sns.boxplot(x="judgment", y="verb", color = 'lightgray', data=df_typmannsp, orient="h", dodge=False)
			# ~ plt.xticks([np.amin(df_long["judgment"].to_numpy()), -1.0, -0.5, 0, 0.5, np.amax(df_long["judgment"].to_numpy())])	# manual
			plt.xticks(np.arange(np.amin(df_long["judgment"].to_numpy()), np.amax(df_long["judgment"].to_numpy())+0.5, 0.5))	# range
			plt.savefig(outdir+'plot_boxplot_'+str(typ)+str(mannsp)+'.png')
			plt.close()

# resumes preprocessing input judgments to compute normalized scores
df_full = df_full.assign(mean=df_full_zscores.mean(axis=1)) # adds mean z-score column
df_full['judg']=(df_full['mean']-df_full['mean'].min())/(df_full['mean'].max()-df_full['mean'].min()) # normalizes z-scores to 0-1 range
print("judgment data preprocessed! (0-1 normalized mean within-subject z-scores) \r\n")
	
# creates sub-dataframes containing only target sentences
df_judg = df_full[df_full.values == 'target']  # WIDE (z-scores)
df_long_target = df_long_raw[df_long_raw.values == 'target']  # LONG (raw judgs, for LMEM)


# this script iterates over 1+ SPS inputs
for sps in os.listdir(args.sps):
	
	if not os.path.exists(args.output+os.path.basename(sps)+'/'):    
		os.makedirs(args.output+os.path.basename(sps)+'/')
	outfolder = args.output+os.path.basename(sps)+'/'
	
	df_sps = pd.read_csv(args.sps+sps, sep=' ', header=None)
	
	# compute z-scores of SPS to make results comparable across models (comment out this paragraph to consider raw SPS scores)
	numeric_cols = df_sps.select_dtypes('float').columns
	df_sps_zscores = df_sps[numeric_cols].apply(zscore) # computes within-subject z-scores
	df_sps = df_sps_zscores.join(df_sps[df_sps.columns.difference(df_sps_zscores.columns)])
	
	df_merge = df_judg.merge(df_sps,left_on='verb', right_on=0) # merge judgments and sps in the same WIDE dataframe
	df = df_merge[["verb", "sentence", "telicity", "perfectivity", "iterativity", "mannspec", "judg", 1]] # subsets df_merge to get relevant columns
	df = df.rename(columns={1: "sps"}) # columns renamed to be more readable
	
	# compute mean judgments regardless of aspectual type (needed below to plot preliminary scatterplot of sps ~ judgments)
	# we do that here because we need sps values to be exactly the same for each aspectual type!
	df_aggr = df.groupby(['verb', 'sentence', 'sps'],as_index=False).agg(lambda x : x.mean() if x.dtype=='float' else x.head(1))
	
	df['sps'] = df['sps'].round(14) # round sps values to 14th decimal place
	df['sps'] = df['sps'].apply(str) # make sps temporarily string
	df.sps=df.sps+df.groupby('sps').cumcount().astype(str) # add ordinal number as 15th decimal place to sps values (to make them unique)
	df['sps'] = df['sps'].apply(float) # (re)make sps a float
	
	# these X and Y values are used to compute the StOT model and also to create a scatterplot
	x_values = df["sps"].to_numpy()
	y_values = df["judg"].to_numpy()
	
	sps_min = np.amin(x_values)
	sps_max = np.amax(x_values)
	
	print("nice "+os.path.basename(sps)+" SPS data you have there")
	
# ----------------------------------------------------------------------------------	
# preliminary stats and plots to see individual and overall contribution of predictors to the actual judgments

	# compute linear mixed-effects model to predict raw judgments from z-scored sps and aspectual features of input
	# yes, even with Likert-scale data! check Kizach (2014) and Bross (2019)
	
	with open(outdir+'lmem_'+os.path.basename(sps)+'.txt', 'w') as file_lmem: 
		df_long_merge = df_long_target.merge(df_sps,left_on='verb', right_on=0) # merge judgments and sps in the same LONG dataframe
		df_long_merge = df_long_merge.rename(columns={1: "sps"}) # columns renamed to be more readable
		# saves input to LMEM in csv file, just in case you are more comfortable interpreting the LMEM in R (lmer + easystats/report)
		df_long_merge.to_csv(outdir+"dataframe_input_lmem_"+os.path.basename(sps)+".csv")
		df_long_merge["group"] = 1  # this line and the next one are necessary to have more than one random effect in the LMEM
		vcf = {"verb": "0 + C(verb)", "subject": "0 + C(subject)"} # equivalent to R's (1|verb)+(1|subject)
		
		# this is the crappiest way EVER to force-set the right reference factors in the LMEM, but it works
		df_long_merge['telicity'] = df_long_merge['telicity'].replace({'telic': '1_telic', 'atelic': '2_atelic'})
		df_long_merge['perfectivity'] = df_long_merge['perfectivity'].replace({'perf': '1_perf', 'imperf': '2_imperf'})
		df_long_merge['iterativity'] = df_long_merge['iterativity'].replace({'noiter': '1_noiter', 'iter': '2_iter'})
		df_long_merge['mannspec'] = df_long_merge['mannspec'].replace({'spec': '1_spec', 'nospec': '2_nospec'})
		
		lmem = sm.MixedLM.from_formula("judgment ~ sps + telicity + perfectivity + iterativity + mannspec", groups="group", vc_formula=vcf, re_formula="0", data=df_long_merge)  
		lmem_fit = lmem.fit()
		file_lmem.write(str(lmem_fit.summary()))
		print("fitted linear mixed-effects model on input data")
		    
		
	# scatterplot: SPS ~ judgments (aspectual types aggregated!)
	plt.figure()
	slope, intercept, r_value, p_value, std_err = linregress(df_aggr['sps'],df_aggr['judg'])  # get coeffs of linear fit
	ax = sns.regplot(x="sps", y="judg", data=df_aggr, color='k', line_kws={'label':"Pearson r={0:.3f}, p = {1:.3f}".format(r_value, p_value)})
	ax.legend() # plot legend with stats
	plt.xlabel("SPS")
	plt.ylabel("average acceptability judgment")
	plt.yticks(np.arange(0, 1.1, 0.1))
	plt.grid(color='grey', linestyle='dotted', linewidth=0.2)
	plt.legend(loc="best")
	plt.savefig(outdir+'plot_preliminary_scatterplot_'+os.path.basename(sps)+'.png')
	plt.close()
	
	# boxplot: telicity ~ judgments
	plt.figure()
	ax = sns.boxplot(x="telicity", y="judg", color = 'lightgray', data=df, dodge=False)
	add_stat_annotation(ax, data=df, x="telicity", y="judg",
				box_pairs=[("telic", "atelic")],
				test='Mann-Whitney', text_format='star', loc='outside', verbose=0) # text_format='star', verbose=2, loc='inside'
	plt.savefig(outdir+'plot_preliminary_boxplot_telicity.png')
	plt.close()		
	
	# boxplot: perfectivity ~ judgments
	plt.figure()
	ax = sns.boxplot(x="perfectivity", y="judg", color = 'lightgray', data=df, dodge=False)
	add_stat_annotation(ax, data=df, x="perfectivity", y="judg",
				box_pairs=[("perf", "imperf")],
				test='Mann-Whitney', text_format='star', loc='outside', verbose=0) # text_format='star', verbose=2, loc='inside'
	plt.savefig(outdir+'plot_preliminary_boxplot_perfectivity.png')
	plt.close()		
	
	# boxplot: iterativity ~ judgments
	plt.figure()
	ax = sns.boxplot(x="iterativity", y="judg", color = 'lightgray', data=df, dodge=False)
	add_stat_annotation(ax, data=df, x="iterativity", y="judg",
				box_pairs=[("iter", "noiter")],
				test='Mann-Whitney', text_format='star', loc='outside', verbose=0) # text_format='star', verbose=2, loc='inside'
	plt.savefig(outdir+'plot_preliminary_boxplot_iterativity.png')
	plt.close()		
	
	# boxplot: mannspec ~ judgments
	plt.figure()
	ax = sns.boxplot(x="mannspec", y="judg", color = 'lightgray', data=df, dodge=False)
	add_stat_annotation(ax, data=df, x="mannspec", y="judg",
				box_pairs=[("spec", "nospec")],
				test='Mann-Whitney', text_format='star', loc='outside', verbose=0) # text_format='star', verbose=2, loc='inside'
	plt.savefig(outdir+'plot_preliminary_boxplot_mannspec.png')
	plt.close()		
	
	print("plotted preliminary plots")
# ----------------------------------------------------------------------------------	


	# functions: probabilities of *INT ARG being ranked over FaithArg, TelicEnd, PerfCoda, Non-IterArg, MannSpecArg
	
	def rerankFaith(x,d1,g1):
		return (((d1-g1)/(sps_max-sps_min))*(x-sps_min))+g1
	
	def rerankTelic(x,d2,g2):
		return (((d2-g2)/(sps_max-sps_min))*(x-sps_min))+g2
		
	def rerankPerf(x,d3,g3):
		return (((d3-g3)/(sps_max-sps_min))*(x-sps_min))+g3
		
	def rerankNonIter(x,d4,g4):
		return (((d4-g4)/(sps_max-sps_min))*(x-sps_min))+g4
		
	def rerankMannSpec(x,d5,g5):
		return (((d5-g5)/(sps_max-sps_min))*(x-sps_min))+g5
		
		
	# functions: 16 individual partial orderings as joint probabilities of independent pairwise orderings
	
	def jointIFTPNM(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5):
		return rerankFaith(x,d1,g1)*rerankTelic(x,d2,g2)*rerankPerf(x,d3,g3)*rerankNonIter(x,d4,g4)*rerankMannSpec(x,d5,g5)
	
	def jointTIFPNM(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5):
		return rerankFaith(x,d1,g1)*(1-rerankTelic(x,d2,g2))*rerankPerf(x,d3,g3)*rerankNonIter(x,d4,g4)*rerankMannSpec(x,d5,g5)
	
	def jointPIFTNM(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5):
		return rerankFaith(x,d1,g1)*rerankTelic(x,d2,g2)*(1-rerankPerf(x,d3,g3))*rerankNonIter(x,d4,g4)*rerankMannSpec(x,d5,g5)
	
	def jointTPIFNM(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5):
		return rerankFaith(x,d1,g1)*(1-rerankTelic(x,d2,g2))*(1-rerankPerf(x,d3,g3))*rerankNonIter(x,d4,g4)*rerankMannSpec(x,d5,g5)
	
	def jointMIFTPN(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5):
		return rerankFaith(x,d1,g1)*rerankTelic(x,d2,g2)*rerankPerf(x,d3,g3)*rerankNonIter(x,d4,g4)*(1-rerankMannSpec(x,d5,g5))
	
	def jointMTIFPN(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5):
		return rerankFaith(x,d1,g1)*(1-rerankTelic(x,d2,g2))*rerankPerf(x,d3,g3)*rerankNonIter(x,d4,g4)*(1-rerankMannSpec(x,d5,g5))
	
	def jointMPIFTN(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5):
		return rerankFaith(x,d1,g1)*rerankTelic(x,d2,g2)*(1-rerankPerf(x,d3,g3))*rerankNonIter(x,d4,g4)*(1-rerankMannSpec(x,d5,g5))
	
	def jointMTPIFN(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5):
		return rerankFaith(x,d1,g1)*(1-rerankTelic(x,d2,g2))*(1-rerankPerf(x,d3,g3))*rerankNonIter(x,d4,g4)*(1-rerankMannSpec(x,d5,g5))
	
	def jointNIFTPM(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5):
		return rerankFaith(x,d1,g1)*rerankTelic(x,d2,g2)*rerankPerf(x,d3,g3)*(1-rerankNonIter(x,d4,g4))*rerankMannSpec(x,d5,g5)
	
	def jointNTIFPM(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5):
		return rerankFaith(x,d1,g1)*(1-rerankTelic(x,d2,g2))*rerankPerf(x,d3,g3)*(1-rerankNonIter(x,d4,g4))*rerankMannSpec(x,d5,g5)
	
	def jointNPIFTM(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5):
		return rerankFaith(x,d1,g1)*rerankTelic(x,d2,g2)*(1-rerankPerf(x,d3,g3))*(1-rerankNonIter(x,d4,g4))*rerankMannSpec(x,d5,g5)
	
	def jointNTPIFM(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5):
		return rerankFaith(x,d1,g1)*(1-rerankTelic(x,d2,g2))*(1-rerankPerf(x,d3,g3))*(1-rerankNonIter(x,d4,g4))*rerankMannSpec(x,d5,g5)
	
	def jointMNIFTP(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5):
		return rerankFaith(x,d1,g1)*rerankTelic(x,d2,g2)*rerankPerf(x,d3,g3)*(1-rerankNonIter(x,d4,g4))*(1-rerankMannSpec(x,d5,g5))
	
	def jointMNTIFP(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5):
		return rerankFaith(x,d1,g1)*(1-rerankTelic(x,d2,g2))*rerankPerf(x,d3,g3)*(1-rerankNonIter(x,d4,g4))*(1-rerankMannSpec(x,d5,g5))
	
	def jointMNPIFT(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5):
		return rerankFaith(x,d1,g1)*rerankTelic(x,d2,g2)*(1-rerankPerf(x,d3,g3))*(1-rerankNonIter(x,d4,g4))*(1-rerankMannSpec(x,d5,g5))
	
	def jointMNTPIF(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5):
		return rerankFaith(x,d1,g1)*(1-rerankTelic(x,d2,g2))*(1-rerankPerf(x,d3,g3))*(1-rerankNonIter(x,d4,g4))*(1-rerankMannSpec(x,d5,g5))
		
		
	# functions: probabilities of the 16 combinations of aspectual types
	
	def probTelPerfIterSpec(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5):
		return jointIFTPNM(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5) + jointNIFTPM(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5)
		
	def probTelPerfNoiterSpec(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5):
		return jointIFTPNM(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5)
		
	def probTelImperfIterSpec(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5):
		return jointIFTPNM(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5) + jointPIFTNM(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5) + jointNIFTPM(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5) + jointNPIFTM(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5)
		
	def probTelImperfNoiterSpec(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5):
		return jointIFTPNM(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5) + jointPIFTNM(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5)
		
		
		
		
	def probAtelPerfIterSpec(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5):
		return jointIFTPNM(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5) + jointTIFPNM(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5) + jointNIFTPM(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5) + jointNTIFPM(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5)
		
	def probAtelPerfNoiterSpec(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5):
		return jointIFTPNM(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5) + jointTIFPNM(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5)
		
	def probAtelImperfIterSpec(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5):
		return jointIFTPNM(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5) + jointTIFPNM(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5) + jointPIFTNM(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5) + jointTPIFNM(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5) + jointNIFTPM(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5) + jointNTIFPM(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5) + jointNPIFTM(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5) + jointNTPIFM(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5)
		
	def probAtelImperfNoiterSpec(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5):
		return jointIFTPNM(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5) + jointTIFPNM(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5) + jointPIFTNM(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5) + jointTPIFNM(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5)
		
	def probTelPerfIterNospec(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5):
		return jointIFTPNM(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5) + jointMIFTPN(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5) + jointNIFTPM(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5) + jointMNIFTP(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5)
		
	def probTelPerfNoiterNospec(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5):
		return jointIFTPNM(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5) + jointMIFTPN(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5)
		
	def probTelImperfIterNospec(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5):
		return jointIFTPNM(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5) + jointPIFTNM(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5) + jointMIFTPN(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5) + jointMPIFTN(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5) + jointNIFTPM(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5) + jointNPIFTM(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5) + jointMNIFTP(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5) + jointMNPIFT(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5)
		
	def probTelImperfNoiterNospec(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5):
		return jointIFTPNM(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5) + jointPIFTNM(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5) + jointMIFTPN(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5) + jointMPIFTN(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5)
		
	def probAtelPerfIterNospec(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5):
		return jointIFTPNM(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5) + jointTIFPNM(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5) + jointMIFTPN(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5) + jointMTIFPN(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5) + jointNIFTPM(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5) + jointNTIFPM(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5) + jointMNIFTP(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5) + jointMNTIFP(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5)
		
	def probAtelPerfNoiterNospec(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5):
		return jointIFTPNM(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5) + jointTIFPNM(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5) + jointMIFTPN(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5) + jointMTIFPN(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5)
		
	def probAtelImperfIterNospec(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5):
		return jointIFTPNM(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5) + jointTIFPNM(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5) + jointPIFTNM(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5) + jointTPIFNM(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5) + jointMIFTPN(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5) + jointMTIFPN(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5) + jointMPIFTN(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5) + jointMTPIFN(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5) + jointNIFTPM(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5) + jointNTIFPM(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5) + jointNPIFTM(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5) + jointNTPIFM(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5) + jointMNIFTP(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5) + jointMNTIFP(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5) + jointMNPIFT(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5) + jointMNTPIF(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5)
		
	def probAtelImperfNoiterNospec(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5):
		return jointIFTPNM(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5) + jointTIFPNM(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5) + jointPIFTNM(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5) + jointTPIFNM(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5) + jointMIFTPN(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5) + jointMTIFPN(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5) + jointMPIFTN(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5) + jointMTPIFN(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5)
	
		
	# subsetting the original dataframe to get a subframe for each aspectual type (used for conditions in funcMedina())
	# I have to come back to this and make it snappier. It's horrible to look at and must be also horrible to run
	
	df_Tel = df[df.values == 'telic']
	df_Atel = df[df.values == 'atelic']
	
	df_TelPerf = df_Tel[df_Tel.values == 'perf']
	df_AtelPerf = df_Atel[df_Atel.values == 'perf']
	df_TelImperf = df_Tel[df_Tel.values == 'imperf']
	df_AtelImperf = df_Atel[df_Atel.values == 'imperf']
	
	df_TelPerfIter = df_TelPerf[df_TelPerf.values == 'iter']
	df_AtelPerfIter = df_AtelPerf[df_AtelPerf.values == 'iter']
	df_TelImperfIter = df_TelImperf[df_TelImperf.values == 'iter']
	df_AtelImperfIter = df_AtelImperf[df_AtelImperf.values == 'iter']
	df_TelPerfNoiter = df_TelPerf[df_TelPerf.values == 'noiter']
	df_AtelPerfNoiter = df_AtelPerf[df_AtelPerf.values == 'noiter']
	df_TelImperfNoiter = df_TelImperf[df_TelImperf.values == 'noiter']
	df_AtelImperfNoiter = df_AtelImperf[df_AtelImperf.values == 'noiter']
	
	df_TelPerfIterSpec = df_TelPerfIter[df_TelPerfIter.values == 'spec']
	df_AtelPerfIterSpec = df_AtelPerfIter[df_AtelPerfIter.values == 'spec']
	df_TelImperfIterSpec = df_TelImperfIter[df_TelImperfIter.values == 'spec']
	df_AtelImperfIterSpec = df_AtelImperfIter[df_AtelImperfIter.values == 'spec']
	df_TelPerfNoiterSpec = df_TelPerfNoiter[df_TelPerfNoiter.values == 'spec']
	df_AtelPerfNoiterSpec = df_AtelPerfNoiter[df_AtelPerfNoiter.values == 'spec']
	df_TelImperfNoiterSpec = df_TelImperfNoiter[df_TelImperfNoiter.values == 'spec']
	df_AtelImperfNoiterSpec = df_AtelImperfNoiter[df_AtelImperfNoiter.values == 'spec']
	df_TelPerfIterNospec = df_TelPerfIter[df_TelPerfIter.values == 'nospec']
	df_AtelPerfIterNospec = df_AtelPerfIter[df_AtelPerfIter.values == 'nospec']
	df_TelImperfIterNospec = df_TelImperfIter[df_TelImperfIter.values == 'nospec']
	df_AtelImperfIterNospec = df_AtelImperfIter[df_AtelImperfIter.values == 'nospec']
	df_TelPerfNoiterNospec = df_TelPerfNoiter[df_TelPerfNoiter.values == 'nospec']
	df_AtelPerfNoiterNospec = df_AtelPerfNoiter[df_AtelPerfNoiter.values == 'nospec']
	df_TelImperfNoiterNospec = df_TelImperfNoiter[df_TelImperfNoiter.values == 'nospec']
	df_AtelImperfNoiterNospec = df_AtelImperfNoiter[df_AtelImperfNoiter.values == 'nospec']
		
	# function to optimize!!! probability of implicit object output across all acceptability judgments
	# np.in1d(value, numpy_array)	yields the same result as the deprecated "value in numpy_array"
	def funcMedina(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5):
		return np.where( # this is necessary because scipy.optimize.curve_fit cannot fit a function whose return depends on ifs
		np.in1d(x, df_TelPerfIterSpec["sps"].to_numpy()), # conditions for TelPerfIterSpec
		probTelPerfIterSpec(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5), #probTelPerfIterSpec
		np.where(
		np.in1d(x, df_TelImperfIterSpec["sps"].to_numpy()), # conditions for TelImperfIterSpec
		probTelImperfIterSpec(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5), #probTelImperfIterSpec
		np.where(
		np.in1d(x, df_AtelPerfIterSpec["sps"].to_numpy()), # conditions for AtelPerfIterSpec
		probAtelPerfIterSpec(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5), #probAtelPerfIterSpec
		np.where(
		np.in1d(x, df_AtelImperfIterSpec["sps"].to_numpy()), # conditions for AtelImperfIterSpec
		probAtelImperfIterSpec(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5), #probAtelImperfIterSpec
		np.where(
		np.in1d(x, df_TelPerfNoiterSpec["sps"].to_numpy()), # conditions for TelPerfNoiterSpec
		probTelPerfNoiterSpec(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5), #probTelPerfNoiterSpec
		np.where(
		np.in1d(x, df_TelImperfNoiterSpec["sps"].to_numpy()), # conditions for TelImperfNoiterSpec
		probTelImperfNoiterSpec(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5), #probTelImperfNoiterSpec
		np.where(
		np.in1d(x, df_AtelPerfNoiterSpec["sps"].to_numpy()), # conditions for AtelPerfNoiterSpec
		probAtelPerfNoiterSpec(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5), #probAtelPerfNoiterSpec
		np.where(
		np.in1d(x, df_AtelImperfNoiterSpec["sps"].to_numpy()), # conditions for AtelImperfNoiterSpec
		probAtelImperfNoiterSpec(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5), #probAtelImperfNoiterSpec
		np.where(
		np.in1d(x, df_TelPerfIterNospec["sps"].to_numpy()), # conditions for TelPerfIterNospec
		probTelPerfIterNospec(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5), #probTelPerfIterNospec
		np.where(
		np.in1d(x, df_TelImperfIterNospec["sps"].to_numpy()), # conditions for TelImperfIterNospec
		probTelImperfIterNospec(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5), #probTelImperfIterNospec
		np.where(
		np.in1d(x, df_AtelPerfIterNospec["sps"].to_numpy()), # conditions for AtelPerfIterNospec
		probAtelPerfIterNospec(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5), #probAtelPerfIterNospec
		np.where(
		np.in1d(x, df_AtelImperfIterNospec["sps"].to_numpy()), # conditions for AtelImperfIterNospec
		probAtelImperfIterNospec(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5), #probAtelImperfIterNospec
		np.where(
		np.in1d(x, df_TelPerfNoiterNospec["sps"].to_numpy()), # conditions for TelPerfNoiterNospec
		probTelPerfNoiterNospec(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5), #probTelPerfNoiterNospec
		np.where(
		np.in1d(x, df_TelImperfNoiterNospec["sps"].to_numpy()), # conditions for TelImperfNoiterNospec
		probTelImperfNoiterNospec(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5), #probTelImperfNoiterNospec
		np.where(
		np.in1d(x, df_AtelPerfNoiterNospec["sps"].to_numpy()), # conditions for AtelPerfNoiterNospec
		probAtelPerfNoiterNospec(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5), #probAtelPerfNoiterNospec
		np.where(
		np.in1d(x, df_AtelImperfNoiterNospec["sps"].to_numpy()), # conditions for AtelImperfNoiterNospec
		probAtelImperfNoiterNospec(x,d1,g1,d2,g2,d3,g3,d4,g4,d5,g5), #probAtelImperfNoiterNospec
		x*1000 # PLACEHOLDER value to return if no condition is True (plz don't do that to me oh mighty Python)
		))
		))
		))
		))
		))
		))
		))
		))
		
	print("defined the functions to model input data in StOT")
		
	# optimize constants for the Medina function (the parameters to be estimated are bounded within [0,1])
	constantsMedina, _ = curve_fit(funcMedina, x_values, y_values, bounds=(0,1.)) # verbose = bounds=((0,0,0,0,0,0),(1.,1.,1.,1.,1.,1.))
	# ~ print('Medina parameters: {}'.format(constantsMedina))
	
	with open(outfolder+'constantsMedina.txt', 'w') as file_constants: 
		file_constants.write("g1: "+str('{0:.3f}'.format(constantsMedina[1]))+"\t d1: "+str('{0:.3f}'.format(constantsMedina[0]))+'\r\n')
		file_constants.write("g2: "+str('{0:.3f}'.format(constantsMedina[3]))+"\t d2: "+str('{0:.3f}'.format(constantsMedina[2]))+'\r\n')
		file_constants.write("g3: "+str('{0:.3f}'.format(constantsMedina[5]))+"\t d3: "+str('{0:.3f}'.format(constantsMedina[4]))+'\r\n')
		file_constants.write("g4: "+str('{0:.3f}'.format(constantsMedina[7]))+"\t d4: "+str('{0:.3f}'.format(constantsMedina[6]))+'\r\n')
		file_constants.write("g5: "+str('{0:.3f}'.format(constantsMedina[9]))+"\t d5: "+str('{0:.3f}'.format(constantsMedina[8]))+'\r\n')
	
	print("model fitted to input data! it's all downhill from here :)")
	
	
	# function to compute individual squared error ('operationally defined to be high if greater than 0.50')
	def calcSquaredError(A,B): 
		dif = A.ravel() - B.ravel()
		return np.dot(dif, dif)
	
	# for the night is dark and full of errors
	with open(outfolder+'errors.txt', 'w') as file_errors: 
		y_pred= []
		list_errors = []
		for el in x_values: # compute individual squared error (comparing actual and predicted judgments)
			y_pred.append(funcMedina(el,*constantsMedina))
			squarederror = calcSquaredError(df[df.values == el]["judg"], funcMedina(el,*constantsMedina))
			list_errors.append(squarederror)
			
			file_errors.write(
			df[df.values == el]["verb"].tolist()[0]+'\t'+ 	# verb
			df[df.values == el]["telicity"].tolist()[0]+'\t'+	# telicity
			df[df.values == el]["perfectivity"].tolist()[0]+'\t'+ 	# perfectivity
			df[df.values == el]["iterativity"].tolist()[0]+'\t'+ 	# iterativity
			df[df.values == el]["mannspec"].tolist()[0]+'\t'+ 	# mannspec
			'SPS: '+ str('{0:.3f}'.format(el))+'\t'+ 
			'real: '+ str('{0:.3f}'.format(float(df[df.values == el]["judg"])))+'\t'+ 
			'pred: '+ str('{0:.3f}'.format(funcMedina(el,*constantsMedina)[0]))+'\t'+ 
			'sq_error: '+ str('{0:.3f}'.format(squarederror)+'\r\n')
			)
			
		
	# compute and print summed squared error (may range between 0 and total number of sentences)
		file_errors.write("\r\n"+"summed squared error: "+ str('{0:.3f}'.format(sum(list_errors)))+'\r\n')	
		print("computed individual and summed squared error")
		
	
	# same as x_values and y_values, but divided into each aspectual type (iterate over the list to create plots)	
	x_asptypes = [df_TelPerfIterSpec["sps"].to_numpy(), df_TelImperfIterSpec["sps"].to_numpy(), df_AtelPerfIterSpec["sps"].to_numpy(), df_AtelImperfIterSpec["sps"].to_numpy(), df_TelPerfNoiterSpec["sps"].to_numpy(), df_TelImperfNoiterSpec["sps"].to_numpy(), df_AtelPerfNoiterSpec["sps"].to_numpy(), df_AtelImperfNoiterSpec["sps"].to_numpy(),df_TelPerfIterNospec["sps"].to_numpy(), df_TelImperfIterNospec["sps"].to_numpy(), df_AtelPerfIterNospec["sps"].to_numpy(), df_AtelImperfIterNospec["sps"].to_numpy(), df_TelPerfNoiterNospec["sps"].to_numpy(), df_TelImperfNoiterNospec["sps"].to_numpy(), df_AtelPerfNoiterNospec["sps"].to_numpy(), df_AtelImperfNoiterNospec["sps"].to_numpy()]
	
	y_asptypes = [df_TelPerfIterSpec["judg"].to_numpy(), df_TelImperfIterSpec["judg"].to_numpy(), df_AtelPerfIterSpec["judg"].to_numpy(), df_AtelImperfIterSpec["judg"].to_numpy(), df_TelPerfNoiterSpec["judg"].to_numpy(), df_TelImperfNoiterSpec["judg"].to_numpy(), df_AtelPerfNoiterSpec["judg"].to_numpy(), df_AtelImperfNoiterSpec["judg"].to_numpy(),df_TelPerfIterNospec["judg"].to_numpy(), df_TelImperfIterNospec["judg"].to_numpy(), df_AtelPerfIterNospec["judg"].to_numpy(), df_AtelImperfIterNospec["judg"].to_numpy(), df_TelPerfNoiterNospec["judg"].to_numpy(), df_TelImperfNoiterNospec["judg"].to_numpy(), df_AtelPerfNoiterNospec["judg"].to_numpy(), df_AtelImperfNoiterNospec["judg"].to_numpy()]
	
	names_asptypes = [
	df_TelPerfIterSpec.telicity.unique()[0]+"_"+df_TelPerfIterSpec.perfectivity.unique()[0]+"_"+df_TelPerfIterSpec.iterativity.unique()[0]+"_"+df_TelPerfIterSpec.mannspec.unique()[0],
	df_TelImperfIterSpec.telicity.unique()[0]+"_"+df_TelImperfIterSpec.perfectivity.unique()[0]+"_"+df_TelImperfIterSpec.iterativity.unique()[0]+"_"+df_TelImperfIterSpec.mannspec.unique()[0],
	df_AtelPerfIterSpec.telicity.unique()[0]+"_"+df_AtelPerfIterSpec.perfectivity.unique()[0]+"_"+df_AtelPerfIterSpec.iterativity.unique()[0]+"_"+df_AtelPerfIterSpec.mannspec.unique()[0],
	df_AtelImperfIterSpec.telicity.unique()[0]+"_"+df_AtelImperfIterSpec.perfectivity.unique()[0]+"_"+df_AtelImperfIterSpec.iterativity.unique()[0]+"_"+df_AtelImperfIterSpec.mannspec.unique()[0],
	df_TelPerfNoiterSpec.telicity.unique()[0]+"_"+df_TelPerfNoiterSpec.perfectivity.unique()[0]+"_"+df_TelPerfNoiterSpec.iterativity.unique()[0]+"_"+df_TelPerfNoiterSpec.mannspec.unique()[0],
	df_TelImperfNoiterSpec.telicity.unique()[0]+"_"+df_TelImperfNoiterSpec.perfectivity.unique()[0]+"_"+df_TelImperfNoiterSpec.iterativity.unique()[0]+"_"+df_TelImperfNoiterSpec.mannspec.unique()[0],
	df_AtelPerfNoiterSpec.telicity.unique()[0]+"_"+df_AtelPerfNoiterSpec.perfectivity.unique()[0]+"_"+df_AtelPerfNoiterSpec.iterativity.unique()[0]+"_"+df_AtelPerfNoiterSpec.mannspec.unique()[0],
	df_AtelImperfNoiterSpec.telicity.unique()[0]+"_"+df_AtelImperfNoiterSpec.perfectivity.unique()[0]+"_"+df_AtelImperfNoiterSpec.iterativity.unique()[0]+"_"+df_AtelImperfNoiterSpec.mannspec.unique()[0],	
	df_TelPerfIterNospec.telicity.unique()[0]+"_"+df_TelPerfIterNospec.perfectivity.unique()[0]+"_"+df_TelPerfIterNospec.iterativity.unique()[0]+"_"+df_TelPerfIterNospec.mannspec.unique()[0],
	df_TelImperfIterNospec.telicity.unique()[0]+"_"+df_TelImperfIterNospec.perfectivity.unique()[0]+"_"+df_TelImperfIterNospec.iterativity.unique()[0]+"_"+df_TelImperfIterNospec.mannspec.unique()[0],
	df_AtelPerfIterNospec.telicity.unique()[0]+"_"+df_AtelPerfIterNospec.perfectivity.unique()[0]+"_"+df_AtelPerfIterNospec.iterativity.unique()[0]+"_"+df_AtelPerfIterNospec.mannspec.unique()[0],
	df_AtelImperfIterNospec.telicity.unique()[0]+"_"+df_AtelImperfIterNospec.perfectivity.unique()[0]+"_"+df_AtelImperfIterNospec.iterativity.unique()[0]+"_"+df_AtelImperfIterNospec.mannspec.unique()[0],
	df_TelPerfNoiterNospec.telicity.unique()[0]+"_"+df_TelPerfNoiterNospec.perfectivity.unique()[0]+"_"+df_TelPerfNoiterNospec.iterativity.unique()[0]+"_"+df_TelPerfNoiterNospec.mannspec.unique()[0],
	df_TelImperfNoiterNospec.telicity.unique()[0]+"_"+df_TelImperfNoiterNospec.perfectivity.unique()[0]+"_"+df_TelImperfNoiterNospec.iterativity.unique()[0]+"_"+df_TelImperfNoiterNospec.mannspec.unique()[0],
	df_AtelPerfNoiterNospec.telicity.unique()[0]+"_"+df_AtelPerfNoiterNospec.perfectivity.unique()[0]+"_"+df_AtelPerfNoiterNospec.iterativity.unique()[0]+"_"+df_AtelPerfNoiterNospec.mannspec.unique()[0],
	df_AtelImperfNoiterNospec.telicity.unique()[0]+"_"+df_AtelImperfNoiterNospec.perfectivity.unique()[0]+"_"+df_AtelImperfNoiterNospec.iterativity.unique()[0]+"_"+df_AtelImperfNoiterNospec.mannspec.unique()[0]
	]
	
	# for each aspectual type:
		# compute actual~predicted Pearson
		# plot actual & predicted judgs against SPS
	with open(outfolder+'pearson.txt', 'w') as file_pearson: 
		for i in range(len(x_asptypes)):
			
			correl = pearsonr(y_asptypes[i], funcMedina(x_asptypes[i],*constantsMedina)) # Pearson correlation (actual ~ predicted values)
			# ~ print("PEARSON", names_asptypes[i], "r= ", '{0:.3f}'.format(correl[0]), "p= ", '{0:.3f}'.format(correl[1]))
			file_pearson.write(names_asptypes[i]+'\t'+"r= "+'{0:.3f}'.format(correl[0])+'\t'+"p= "+'{0:.3f}'.format(correl[1])+'\r\n')
			
			plt.figure()
			plt.plot(x_asptypes[i],y_asptypes[i],'ko',label='actual judgments')
			plt.plot(x_asptypes[i],funcMedina(x_asptypes[i],*constantsMedina), linestyle='dotted',color='black', label='model for '+str(names_asptypes[i])+' verbs')
			plt.plot(x_asptypes[i],funcMedina(x_asptypes[i],*constantsMedina), markeredgecolor='0', markerfacecolor='0.6', marker='s', linestyle='None', label='predicted judgments')
			plt.xlabel("SPS")
			plt.ylabel("mean acceptability")
			plt.xticks([sps_min, sps_max])	# forces x axis to show every possible value (comment if plot appears too empty)
			plt.yticks(np.arange(0, 1.1, 0.1)) # forces y axis to show every possible value (comment if plot appears too empty)
			plt.grid(color='grey', linestyle='dotted', linewidth=0.2)
			plt.legend(loc="best")
			# ~ plt.show()	
			plt.savefig(outfolder+'plot_actualpred_'+str(names_asptypes[i])+'.png')
			plt.close()
		print("computed Pearson between actual and predicted values")
		print("plotted both actual and predicted values against SPS")
	
	
	# plot prob of the implicit object output for each aspectual_type input across hypothetical SPS values ranging from min to max values
	hypo_x=np.linspace(x_values.min(),x_values.max(),50) # Return evenly spaced numbers over a specified interval. (to create plots)
	plt.figure()
	plt.plot(hypo_x, probTelPerfIterSpec(hypo_x,*constantsMedina), 'r-', label='telic perfective iterative spec')
	plt.plot(hypo_x, probTelImperfIterSpec(hypo_x,*constantsMedina), 'b-', label='telic imperfective iterative spec')
	plt.plot(hypo_x, probAtelPerfIterSpec(hypo_x,*constantsMedina), 'g-', label='atelic perfective iterative spec')
	plt.plot(hypo_x, probAtelImperfIterSpec(hypo_x,*constantsMedina), 'k-', label='atelic imperfective iterative spec')
	plt.plot(hypo_x, probTelPerfNoiterSpec(hypo_x,*constantsMedina), 'r--', label='telic perfective non-iterative spec')
	plt.plot(hypo_x, probTelImperfNoiterSpec(hypo_x,*constantsMedina), 'b--', label='telic imperfective non-iterative spec')
	plt.plot(hypo_x, probAtelPerfNoiterSpec(hypo_x,*constantsMedina), 'g--', label='atelic perfective non-iterative spec')
	plt.plot(hypo_x, probAtelImperfNoiterSpec(hypo_x,*constantsMedina), 'k--', label='atelic imperfective non-iterative spec')
	plt.plot(hypo_x, probTelPerfIterNospec(hypo_x,*constantsMedina), 'r-.', label='telic perfective iterative nonspec')
	plt.plot(hypo_x, probTelImperfIterNospec(hypo_x,*constantsMedina), 'b-.', label='telic imperfective iterative nonspec')
	plt.plot(hypo_x, probAtelPerfIterNospec(hypo_x,*constantsMedina), 'g-.', label='atelic perfective iterative nonspec')
	plt.plot(hypo_x, probAtelImperfIterNospec(hypo_x,*constantsMedina), 'k-.', label='atelic imperfective iterative nonspec')
	plt.plot(hypo_x, probTelPerfNoiterNospec(hypo_x,*constantsMedina), 'r:', label='telic perfective non-iterative nonspec')
	plt.plot(hypo_x, probTelImperfNoiterNospec(hypo_x,*constantsMedina), 'b:', label='telic imperfective non-iterative nonspec')
	plt.plot(hypo_x, probAtelPerfNoiterNospec(hypo_x,*constantsMedina), 'g:', label='atelic perfective non-iterative nonspec')
	plt.plot(hypo_x, probAtelImperfNoiterNospec(hypo_x,*constantsMedina), 'k:', label='atelic imperfective non-iterative nonspec')
	plt.xlabel("SPS")
	plt.ylabel("probability of implicit object output")
	plt.yticks(np.arange(0, 1.1, 0.1))
	plt.grid(color='grey', linestyle='dotted', linewidth=0.2)
	# ~ plt.legend(loc="best")
	plt.legend(loc='center left', bbox_to_anchor=(1, 0))
	# ~ plt.show()
	# ~ plt.savefig(outfolder+'plot_prob_aspectualtypes.png')
	plt.savefig(outfolder+'plot_prob_aspectualtypes.png', bbox_inches='tight')
	plt.close()
	
	
	# create function to plot reranking probabilities iteratively
	def plot_prob_ranking(data):
		outplot = plt.plot([sps_min, sps_max], data, 'k-', marker='o')
		plt.xlabel("SPS")
		plt.xticks([sps_min, sps_max])
		plt.yticks(np.arange(0, 1.1, 0.1))
		plt.grid(color='grey', linestyle='dotted', linewidth=0.2, axis='y')
		return outplot
	
	# plot p (*INT ARG >> FAITH ARG) given sps_min/max on X axis and d1/g1 on Y axis    
	plt.figure()
	plot_prob_ranking([constantsMedina[1], constantsMedina[0]])
	plt.ylabel("p (*INT ARG >> FAITH ARG)")
	plt.savefig(outfolder+'plot_prob_faitharg.png')
	plt.close()
	
	
	# plot p (*INT ARG >> TELIC END) given sps_min/max on X axis and d1/g1 on Y axis
	plt.figure()
	plot_prob_ranking([constantsMedina[3], constantsMedina[2]])
	plt.ylabel("p (*INT ARG >> TELIC END)")
	plt.savefig(outfolder+'plot_prob_telicend.png')
	plt.close()
	
	# plot p (*INT ARG >> PERF CODA) given sps_min/max on X axis and d1/g1 on Y axis
	plt.figure()
	plot_prob_ranking([constantsMedina[5], constantsMedina[4]])
	plt.ylabel("p (*INT ARG >> PERF CODA)")
	plt.savefig(outfolder+'plot_prob_perfcoda.png')
	plt.close()
	
	# plot p (*INT ARG >> NON-ITER ARG) given sps_min/max on X axis and d1/g1 on Y axis
	plt.figure()
	plot_prob_ranking([constantsMedina[7], constantsMedina[6]])
	plt.ylabel("p (*INT ARG >> NON-ITER ARG)")
	plt.savefig(outfolder+'plot_prob_noniterarg.png')
	plt.close()
	
	# plot p (*INT ARG >> MANN SPEC ARG) given sps_min/max on X axis and d1/g1 on Y axis
	plt.figure()
	plot_prob_ranking([constantsMedina[9], constantsMedina[8]])
	plt.ylabel("p (*INT ARG >> MANN SPEC ARG)")
	plt.savefig(outfolder+'plot_prob_mannspecarg.png')
	plt.close()
	
	print("done working with "+os.path.basename(sps)+"\r\n")
print("done!")
