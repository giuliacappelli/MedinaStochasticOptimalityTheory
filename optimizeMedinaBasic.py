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
import tikzplotlib # to convert Python plots to LaTeX

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
                       help='file containing tab-separated raw judgments') # Medina's constraints

my_parser.add_argument('--zipf',
                       '-z',
                       action='store',
                       default='input/judgments/zipf.csv',
                       help='file containing space-separated Zipf values (log transform of frequency per billion words)')

my_parser.add_argument('--output',
                       '-o',
                       action='store',
                       default='output/',
                       help='output folder')

args = my_parser.parse_args()

print("fasten your seatbelts! \r\n")

def label_point(x, y, val, ax):
	a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
	for i, point in a.iterrows():
			ax.text(point['x']+.02, point['y'], str(point['val']))		# dense labels
			# ax.text(point['x']+2.5, point['y']+.025, str(point['val']))		# sparse labels

# create output folders if they do not exist yet
if not os.path.exists(args.output):
    os.makedirs(args.output)

df_full = pd.read_csv(args.judgments, sep='\t') # file has headers!
df_zipf = pd.read_csv(args.zipf, sep=' ', header = None, names=['verb','zipf'])

# ~ # checkpoint! your input dataframe may contain more columns than necessary for the model created by this script
# ~ # if so, the script will remove them
# ~ # if not, your input will stay the same
if 'mannspec' in df_full.columns:
	df_full = df_full.drop(columns=['mannspec'])
if 'iterativity' in df_full.columns:
	df_full = df_full.drop(columns=['iterativity'])
if not ('iterativity' in df_full.columns or 'mannspec' in df_full.columns):
	df_full = df_full

numeric_cols = df_full.select_dtypes('int64').columns
df_full_zscores = df_full[numeric_cols].apply(zscore) # computes within-subject z-scores
print("preprocessing input data... \r\n")

# create temporary long dataframe with z-scores to plot preliminary density plots
df_zscores = df_full_zscores.join(df_full[df_full.columns.difference(df_full_zscores.columns)])
df_long = df_zscores.melt(id_vars=['perfectivity','sentence','telicity','verb'], var_name="subject", value_name="judgment") # long df (z-scores)
df_long_raw = df_full.melt(id_vars=['perfectivity','sentence','telicity','verb'], var_name="subject", value_name="judgment") # long df (raw judgs)

# print(df_long)
df_long.to_csv(args.output+"df_long_zscores"+".csv")
df_long_raw.to_csv(args.output+"df_long_raw"+".csv")

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
		tikzplotlib.save(outdir+'latex_boxplot_'+str(typ)+'.tex')
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
			tikzplotlib.save(outdir+'latex_boxplot_'+str(typ)+str(prf)+'.tex')
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
	sps_name = os.path.basename(sps).replace("_", " ")

	if not os.path.exists(args.output+os.path.basename(sps)+'/'):
		os.makedirs(args.output+os.path.basename(sps)+'/')
	outfolder = args.output+os.path.basename(sps)+'/'

	# df_sps = pd.read_csv(args.sps+sps, sep=' ', header=None)
	df_sps = pd.read_csv(args.sps+sps, sep=' ', header=None, names=['verb','sps'])
	print("nice "+sps_name+" data you have there")

	# Zipf*PISA values, in order to account for frequency effects
	# comment out this paragraph to only consider un-Zipfed PISA values
	df_sps_zipf = pd.merge(df_sps, df_zipf, on='verb') # merge SPS and Zipf dataframes
	df_sps_zipf['zipfed'] = df_sps_zipf['sps'] * df_sps_zipf['zipf'] # computes Zipf*SPS values
	df_sps = df_sps_zipf # renames dataframe to original one (to use below without altering the script)
	df_sps = df_sps.drop(columns=['sps', 'zipf']) # removes now-useless columns
	df_sps=df_sps.rename(columns={col: "" for col in df_sps}) # removes headers in the worst way possible
	df_sps.to_csv(outfolder+os.path.basename(sps)+".zipfed", index=False, index_label=False, header=False, sep=' ') # saving-and-reopening csv cuz am not proficient in pandas
	df_sps = pd.read_csv(outfolder+os.path.basename(sps)+".zipfed", sep=' ', header=None)
	print("SPS*Zipf values computed!")

	# compute z-scores of SPS to make results comparable across models (comment out this paragraph to consider raw SPS scores)
	numeric_cols = df_sps.select_dtypes('float').columns
	df_sps_zscores = df_sps[numeric_cols].apply(zscore)
	df_sps = df_sps_zscores.join(df_sps[df_sps.columns.difference(df_sps_zscores.columns)])

	df_merge = df_judg.merge(df_sps,left_on='verb', right_on=0) # merge judgments and sps in the same WIDE dataframe
	df = df_merge[["verb", "sentence", "telicity", "perfectivity", "judg", 1]] # subsets df_merge to get relevant columns
	df = df.rename(columns={1: "sps"}) # columns renamed to be more readable
	df['sps']=(df['sps']-df['sps'].min())/(df['sps'].max()-df['sps'].min())+0.000000001 # normalizes z-scores to 0-1 range

	# compute mean judgments regardless of aspectual type (needed below to plot preliminary scatterplot of sps ~ judgments)
	# we do that here because we need sps values to be exactly the same for each aspectual type!
	df_aggr = df.groupby(['verb', 'sentence', 'sps'],as_index=False).agg(lambda x : x.mean() if x.dtype=='float' else x.head(1))

	df_aggr.to_csv(outfolder+"df_regression_"+os.path.basename(sps)+".csv")

	df['sps'] = df['sps'].round(14) # round sps values to 14th decimal place
	df['sps'] = df['sps'].apply(str) # make sps temporarily string
	df.sps=df.sps+df.groupby('sps').cumcount().astype(str) # add ordinal number as 15th decimal place to sps values (to make them unique)
	df['sps'] = df['sps'].apply(float) # (re)make sps a float

	# these X and Y values are used to compute the StOT model and also to create a scatterplot
	x_values = df["sps"].to_numpy()
	y_values = df["judg"].to_numpy()

	sps_min = np.amin(x_values)
	sps_max = np.amax(x_values)

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

		lmem = sm.MixedLM.from_formula("judgment ~ sps + telicity + perfectivity", groups="group", vc_formula=vcf, re_formula="0", data=df_long_merge)
		lmem_fit = lmem.fit()
		file_lmem.write(str(lmem_fit.summary()))
		print("fitted linear mixed-effects model on input data")


	# scatterplot: SPS ~ judgments (aspectual types aggregated!)
	plt.figure()
	slope, intercept, r_value, p_value, std_err = linregress(df_aggr['sps'],df_aggr['judg'])  # get coeffs of linear fit
	g = sns.lmplot(x="sps", y="judg", data=df_aggr, fit_reg=False)
	# g._legend.remove()
	sns.regplot(x="sps", y="judg", data=df_aggr, scatter=False, ax=g.axes[0, 0], label="Pearson r={0:.3f}, p = {1:.3f}".format(r_value, p_value)).legend(loc="best")
	plt.xlabel(sps_name)
	plt.ylabel("average acceptability judgment")
	# plt.yticks(np.arange(0, 0.5, 0.1))
	plt.ylim(0,1)
	# plt.xlim(-2,3)
	plt.xlim(0,1)
	plt.grid(color='grey', linestyle='dotted', linewidth=0.2)
	label_point(df_aggr["sps"], df_aggr["judg"], df_aggr["verb"], plt.gca())  # comment to remove labels on scatterplot dots, requires function defined above!
	plt.savefig(outdir+'plot_preliminary_scatterplot_'+os.path.basename(sps)+'.png', dpi=200, bbox_inches='tight') # bbox serve per non far croppare il grafico (mistero sul motivo)
	tikzplotlib.save(outdir+'latex_preliminary_scatterplot_'+os.path.basename(sps)+'.tex')
	plt.close()

	# boxplot: telicity ~ judgments
	plt.figure()
	ax = sns.boxplot(x="telicity", y="judg", color = 'lightgray', data=df, dodge=False)
	add_stat_annotation(ax, data=df, x="telicity", y="judg",
				box_pairs=[("telic", "atelic")],
				test='Mann-Whitney', text_format='star', loc='outside', verbose=0) # text_format='star', verbose=2, loc='inside'
	plt.savefig(outdir+'plot_preliminary_boxplot_telicity.png')
	tikzplotlib.save(outdir+'latex_preliminary_boxplot_telicity.tex')
	plt.close()

	# boxplot: perfectivity ~ judgments
	plt.figure()
	ax = sns.boxplot(x="perfectivity", y="judg", color = 'lightgray', data=df, dodge=False)
	add_stat_annotation(ax, data=df, x="perfectivity", y="judg",
				box_pairs=[("perf", "imperf")],
				test='Mann-Whitney', text_format='star', loc='outside', verbose=0) # text_format='star', verbose=2, loc='inside'
	plt.savefig(outdir+'plot_preliminary_boxplot_perfectivity.png')
	tikzplotlib.save(outdir+'latex_preliminary_boxplot_perfectivity.tex')
	plt.close()

	print("plotted preliminary plots")
# ----------------------------------------------------------------------------------


	# functions: probabilities of *INT ARG being ranked over FaithArg, TelicEnd, PerfCoda

	def rerankFaith(x,d1,g1):
		return (((d1-g1)/(sps_max-sps_min))*(x-sps_min))+g1

	def rerankTelic(x,d2,g2):
		return (((d2-g2)/(sps_max-sps_min))*(x-sps_min))+g2

	def rerankPerf(x,d3,g3):
		return (((d3-g3)/(sps_max-sps_min))*(x-sps_min))+g3


	# functions: 4 individual partial orderings as joint probabilities of independent pairwise orderings
	# Medina (2007: 101) says 8, but the useful ones are just half! namely, the ones where Faith is not above IntArg

	def jointIFTP(x,d1,g1,d2,g2,d3,g3):
		return rerankFaith(x,d1,g1)*rerankTelic(x,d2,g2)*rerankPerf(x,d3,g3)

	def jointTIFP(x,d1,g1,d2,g2,d3,g3):
		return rerankFaith(x,d1,g1)*(1-rerankTelic(x,d2,g2))*rerankPerf(x,d3,g3)

	def jointPIFT(x,d1,g1,d2,g2,d3,g3):
		return rerankFaith(x,d1,g1)*rerankTelic(x,d2,g2)*(1-rerankPerf(x,d3,g3))

	def jointTPIF(x,d1,g1,d2,g2,d3,g3):
		return rerankFaith(x,d1,g1)*(1-rerankTelic(x,d2,g2))*(1-rerankPerf(x,d3,g3))


	# functions: probabilities of the 4 combinations of aspectual types

	def probTelPerf(x,d1,g1,d2,g2,d3,g3):
		return jointIFTP(x,d1,g1,d2,g2,d3,g3)

	def probTelImperf(x,d1,g1,d2,g2,d3,g3):
		return jointIFTP(x,d1,g1,d2,g2,d3,g3) + jointPIFT(x,d1,g1,d2,g2,d3,g3)

	def probAtelPerf(x,d1,g1,d2,g2,d3,g3):
		return jointIFTP(x,d1,g1,d2,g2,d3,g3) + jointTIFP(x,d1,g1,d2,g2,d3,g3)

	def probAtelImperf(x,d1,g1,d2,g2,d3,g3):
		return jointIFTP(x,d1,g1,d2,g2,d3,g3) + jointPIFT(x,d1,g1,d2,g2,d3,g3) + jointTIFP(x,d1,g1,d2,g2,d3,g3) + jointTPIF(x,d1,g1,d2,g2,d3,g3)


	# subsetting the original dataframe to get a subframe for each aspectual type (used for conditions in funcMedina())
	df_Tel = df[df.values == 'telic']
	df_Atel = df[df.values == 'atelic']

	df_Perf = df[df.values == 'perf']
	df_Imperf = df[df.values == 'imperf']

	df_TelPerf = df_Tel[df_Tel.values == 'perf']
	df_AtelPerf = df_Atel[df_Atel.values == 'perf']
	df_TelImperf = df_Tel[df_Tel.values == 'imperf']
	df_AtelImperf = df_Atel[df_Atel.values == 'imperf']

	# function to optimize!!! probability of implicit object output across all acceptability judgments
	# np.in1d(value, numpy_array)	yields the same result as the deprecated "value in numpy_array"
	def funcMedina(x,d1,g1,d2,g2,d3,g3):
		return np.where( # this is necessary because scipy.optimize.curve_fit cannot fit a function whose return depends on ifs
		np.in1d(x, df_TelPerf["sps"].to_numpy()), # conditions for TelPerf
		probTelPerf(x,d1,g1,d2,g2,d3,g3), #probTelPerf
		np.where(
		np.in1d(x, df_TelImperf["sps"].to_numpy()), # conditions for TelImperf
		probTelImperf(x,d1,g1,d2,g2,d3,g3), #probTelImperf
		np.where(
		np.in1d(x, df_AtelPerf["sps"].to_numpy()), # conditions for AtelPerf
		probAtelPerf(x,d1,g1,d2,g2,d3,g3), #probAtelPerf
		np.where(
		np.in1d(x, df_AtelImperf["sps"].to_numpy()), # conditions for AtelImperf
		probAtelImperf(x,d1,g1,d2,g2,d3,g3), #probAtelImperf
		x*1000 # PLACEHOLDER value to return if no condition is True (plz don't do that to me oh mighty Python)
		)
		)
		)
		)

	print("defined the functions to model input data in StOT")

	# optimize constants for the Medina function (the parameters to be estimated are bounded within [0,1])
	constantsMedina, _ = curve_fit(funcMedina, x_values, y_values, bounds=(0,1.)) # verbose = bounds=((0,0,0,0,0,0),(1.,1.,1.,1.,1.,1.))
	# ~ print('Medina parameters: {}'.format(constantsMedina))

	with open(outfolder+'constantsMedina.txt', 'w') as file_constants:
		file_constants.write("g1: "+str('{0:.3f}'.format(constantsMedina[1]))+"\t d1: "+str('{0:.3f}'.format(constantsMedina[0]))+'\r\n')
		file_constants.write("g2: "+str('{0:.3f}'.format(constantsMedina[3]))+"\t d2: "+str('{0:.3f}'.format(constantsMedina[2]))+'\r\n')
		file_constants.write("g3: "+str('{0:.3f}'.format(constantsMedina[5]))+"\t d3: "+str('{0:.3f}'.format(constantsMedina[4]))+'\r\n')

	print("model fitted to input data! it's all downhill from here :)")


	# function to compute individual squared error ('operationally defined to be high if greater than 0.50')
	def calcSquaredError(A,B):
		dif = A.ravel() - B.ravel()
		return np.dot(dif, dif)

	# for the night is dark and full of errors
	with open(outfolder+'errors.txt', 'w') as file_errors:
		mean_actual = df["judg"].mean() # mean of actual values (NOT predicted)
		y_pred= []
		list_errors = []
		list_sum = []
		for el in x_values: # compute individual squared error (comparing actual and predicted judgments)
			y_pred.append(funcMedina(el,*constantsMedina))
			squarederror = calcSquaredError(df[df.values == el]["judg"], funcMedina(el,*constantsMedina))
			squaresum = (df[df.values == el]["judg"].to_numpy()-mean_actual)*(df[df.values == el]["judg"].to_numpy()-mean_actual)
			list_errors.append(squarederror)
			list_sum.append(squaresum[0])

			file_errors.write(
			df[df.values == el]["verb"].tolist()[0]+'\t'+ 	# verb
			df[df.values == el]["telicity"].tolist()[0]+'\t'+	# telicity
			df[df.values == el]["perfectivity"].tolist()[0]+'\t'+ 	# perfectivity
			'SPS: '+ str('{0:.3f}'.format(el))+'\t'+
			'real: '+ str('{0:.3f}'.format(float(df[df.values == el]["judg"])))+'\t'+
			'pred: '+ str('{0:.3f}'.format(funcMedina(el,*constantsMedina)[0]))+'\t'+
			'sq_error: '+ str('{0:.3f}'.format(squarederror)+'\r\n')
			)


		SSE = sum(list_errors) # summed squared error
		SST = sum(list_sum) # total sum of squares
		rsquared = 1 - (SSE/SST)

		n = len(df["judg"].to_numpy()) # number of points in data sample
		k = 3 # number of variables in the model (independent regressors): sps, tel, perf
		adjrsquared = 1 - ( ( (1-rsquared) * (n-1) ) / (n-k-1) )

	# compute and print summed squared error (may range between 0 and total number of sentences)
		file_errors.write("\r\n"+"summed squared error: "+ str('{0:.3f}'.format(SSE))+'\r\n')
		file_errors.write("\r\n"+"total sum of squares: "+ str('{0:.3f}'.format(SST))+'\r\n')
		file_errors.write("\r\n"+"n of points in data sample: "+ str(n)+'\r\n')
		file_errors.write("\r\n"+"n of variables in model: "+ str(k)+'\r\n')
		file_errors.write("\r\n"+"R squared: "+ str('{0:.3f}'.format(rsquared))+'\r\n')
		file_errors.write("\r\n"+"adjusted R squared: "+ str('{0:.3f}'.format(adjrsquared))+'\r\n')
		print("computed individual and summed squared error")


	# same as x_values and y_values, but divided into each aspectual type (iterate over the list to create plots)
	x_asptypes = [df_Tel["sps"].to_numpy(), df_Atel["sps"].to_numpy(), df_Perf["sps"].to_numpy(), df_Imperf["sps"].to_numpy(), df_TelPerf["sps"].to_numpy(), df_TelImperf["sps"].to_numpy(), df_AtelPerf["sps"].to_numpy(), df_AtelImperf["sps"].to_numpy()]
	y_asptypes = [df_Tel["judg"].to_numpy(), df_Atel["judg"].to_numpy(), df_Perf["judg"].to_numpy(), df_Imperf["judg"].to_numpy(), df_TelPerf["judg"].to_numpy(), df_TelImperf["judg"].to_numpy(), df_AtelPerf["judg"].to_numpy(), df_AtelImperf["judg"].to_numpy()]

	names_asptypes = [
	df_Tel.telicity.unique()[0], df_Atel.telicity.unique()[0],
	df_Perf.perfectivity.unique()[0], df_Imperf.perfectivity.unique()[0],
	df_TelPerf.telicity.unique()[0]+"_"+df_TelPerf.perfectivity.unique()[0],
	df_TelImperf.telicity.unique()[0]+"_"+df_TelImperf.perfectivity.unique()[0],
	df_AtelPerf.telicity.unique()[0]+"_"+df_AtelPerf.perfectivity.unique()[0],
	df_AtelImperf.telicity.unique()[0]+"_"+df_AtelImperf.perfectivity.unique()[0]
	]

	# for each aspectual type:
		# compute actual~predicted Pearson
		# plot actual & predicted judgs against SPS
	with open(outfolder+'pearson.txt', 'w') as file_pearson:

		correl = pearsonr(df["judg"].to_numpy(), funcMedina(df["sps"].to_numpy(),*constantsMedina)) # Pearson correlation (actual ~ predicted values)
		file_pearson.write('total Pearson'+'\t'+"r= "+'{0:.3f}'.format(correl[0])+'\t'+"p= "+'{0:.3f}'.format(correl[1])+'\r\n')

		for i in range(len(x_asptypes)):

			correl = pearsonr(y_asptypes[i], funcMedina(x_asptypes[i],*constantsMedina)) # Pearson correlation (actual ~ predicted values)
			file_pearson.write(names_asptypes[i]+'\t'+"r= "+'{0:.3f}'.format(correl[0])+'\t'+"p= "+'{0:.3f}'.format(correl[1])+'\r\n')

			plt.figure()
			plt.plot(x_asptypes[i],y_asptypes[i],'ko',label='actual judgments')
			plt.plot(x_asptypes[i],funcMedina(x_asptypes[i],*constantsMedina), linestyle='dotted',color='black', label='model for '+str(names_asptypes[i])+' verbs')
			plt.plot(x_asptypes[i],funcMedina(x_asptypes[i],*constantsMedina), markeredgecolor='0', markerfacecolor='0.6', marker='s', linestyle='None', label='predicted judgments')
			plt.xlabel(sps_name)
			plt.ylabel("mean acceptability")
			plt.xticks([sps_min, sps_max])	# forces x axis to show every possible value (comment if plot appears too empty)
			plt.yticks(np.arange(0, 1.1, 0.1)) # forces y axis to show every possible value (comment if plot appears too empty)
			plt.grid(color='grey', linestyle='dotted', linewidth=0.2)
			plt.legend(loc="best")
			# ~ plt.show()
			plt.savefig(outfolder+'plot_actualpred_'+str(names_asptypes[i])+'.png')
			tikzplotlib.save(outfolder+'latex_actualpred_'+str(names_asptypes[i])+'.tex')
			plt.close()
		print("computed Pearson between actual and predicted values")
		print("plotted both actual and predicted values against SPS")


	# plot prob of the implicit object output for each aspectual_type input across hypothetical SPS values ranging from min to max values
	hypo_x=np.linspace(x_values.min(),x_values.max(),50) # Return evenly spaced numbers over a specified interval. (to create plots)
	plt.figure()
	plt.plot(hypo_x, probTelPerf(hypo_x,*constantsMedina), 'r-', label='telic perfective')
	plt.plot(hypo_x, probTelImperf(hypo_x,*constantsMedina), 'b-', label='telic imperfective')
	plt.plot(hypo_x, probAtelPerf(hypo_x,*constantsMedina), 'g-', label='atelic perfective')
	plt.plot(hypo_x, probAtelImperf(hypo_x,*constantsMedina), 'k-', label='atelic imperfective')
	plt.xlabel(sps_name)
	plt.ylabel("probability of implicit object output")
	plt.yticks(np.arange(0, 1.1, 0.1))
	plt.grid(color='grey', linestyle='dotted', linewidth=0.2)
	plt.legend(loc="best")
	# ~ plt.show()
	plt.savefig(outfolder+'plot_prob_aspectualtypes.png')
	tikzplotlib.save(outfolder+'latex_prob_aspectualtypes.tex')
	plt.close()


	# create function to plot reranking probabilities iteratively
	def plot_prob_ranking(data):
		outplot = plt.plot([sps_min, sps_max], data, 'k-', marker='o')
		plt.xlabel(sps_name)
		plt.xticks([sps_min, sps_max])
		plt.yticks(np.arange(0, 1.1, 0.1))
		plt.grid(color='grey', linestyle='dotted', linewidth=0.2, axis='y')
		return outplot

	# plot p (*INT ARG >> FAITH ARG) given sps_min/max on X axis and d1/g1 on Y axis
	plt.figure()
	plot_prob_ranking([constantsMedina[1], constantsMedina[0]])
	plt.ylabel("p (*INT ARG >> FAITH ARG)")
	plt.savefig(outfolder+'plot_prob_faitharg.png')
	tikzplotlib.save(outfolder+'latex_prob_faitharg.tex')
	plt.close()

	# plot p (*INT ARG >> TELIC END) given sps_min/max on X axis and d1/g1 on Y axis
	plt.figure()
	plot_prob_ranking([constantsMedina[3], constantsMedina[2]])
	plt.ylabel("p (*INT ARG >> TELIC END)")
	plt.savefig(outfolder+'plot_prob_telicend.png')
	tikzplotlib.save(outfolder+'latex_prob_telicend.tex')
	plt.close()

	# plot p (*INT ARG >> PERF CODA) given sps_min/max on X axis and d1/g1 on Y axis
	plt.figure()
	plot_prob_ranking([constantsMedina[5], constantsMedina[4]])
	plt.ylabel("p (*INT ARG >> PERF CODA)")
	plt.savefig(outfolder+'plot_prob_perfcoda.png')
	tikzplotlib.save(outfolder+'latex_prob_perfcoda.tex')
	plt.close()

	# plot p (*INT ARG >> all other constraints) given sps_min/max on X axis and d1/g1 on Y axis
	plt.figure()
	plt.plot([sps_min, sps_max], [constantsMedina[1], constantsMedina[0]], 'r-', label='p (*INT ARG >> FAITH ARG)')
	plt.plot([sps_min, sps_max], [constantsMedina[3], constantsMedina[2]], 'b-', label='p (*INT ARG >> TELIC END)')
	plt.plot([sps_min, sps_max], [constantsMedina[5], constantsMedina[4]], 'g-', label='p (*INT ARG >> PERF CODA)')
	plt.xlabel(sps_name)
	plt.xticks([sps_min, sps_max])
	plt.yticks(np.arange(0, 1.1, 0.1))
	plt.grid(color='grey', linestyle='dotted', linewidth=0.2, axis='y')
	plt.legend(loc="best")
	plt.ylabel("p (*INT ARG >> other constraints)")
	plt.savefig(outfolder+'plot_prob_alltogether.png')
	tikzplotlib.save(outfolder+'latex_prob_alltogether.tex')
	plt.close()

	print("done working with "+sps_name+"\r\n")
print("done!")
