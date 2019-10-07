import pandas
import os
import glob
#  import pandas as pd
import dask
import dask.dataframe as dd

path = "/home/twolf/scratch-midway2"

#  pattern = "Part*_of_50_cache_before_cuts_SR1_Rn220_16-09-2019.hdf"
pattern = "Part*_of_450_cache_before_cuts_SR1_Bkg_17-09-2019.hdf"

glob_to_find = os.path.join(path, pattern)

filelist = glob.glob(glob_to_find)

CutsToApply = [
  "CutDAQVeto == True",
  "CutFlash == True",
  "CutMuonVeto == True",
  "CutS2Threshold == True",
  "CutS2SingleScatterHE == True",
  "CutS1AreaFractionTopHE == True",
  "CutPosDiffHE == True",
  "-92.9 < z_3d_nn_tf & z_3d_nn_tf < -9 & r_3d_nn_tf < 36.94",

  #"CutCS2AreaFractionTop == True",
  #"CutS2PatternLikelihood == True",
  #"CutS1AreaFractionTop == True",
  #"CutS1MaxPMT == True",
  #"CutS2Tails == True",
  #"CutInteractionPeaksBiggest == True",
]
combined_cut = ""
for cut in CutsToApply:
    print("Adding cut: ", cut)
    combined_cut += "( " + cut + " ) & "
combined_cut = combined_cut[:-3]
print("combined_cut = ", combined_cut)


l_dfs = []
for this_file in filelist:
    print("Reading", this_file)
    df = dd.read_hdf(this_file, key='df_raw')
    df = df.query(combined_cut)
    l_dfs.append(df)

print("Combining dask-dataframes")
master_df = dd.concat(l_dfs)
filename = os.path.join(path, "cache_after_cuts_SR1_Bkg_17-09-2019.hdf")
master_df.to_hdf(filename, key='df_raw', format='table')
