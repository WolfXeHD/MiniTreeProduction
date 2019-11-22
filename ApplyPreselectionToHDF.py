import pandas
import os
import glob
#  import pandas as pd
import dask
import dask.dataframe as dd

import argparse
import sys

def parse_args(args):
    parser = argparse.ArgumentParser(description='Load data from processed minitrees.')
    parser.add_argument('--pattern', type=str, required=True, help="Pattern to find the files to merge.")
    parser.add_argument('--target_file', type=str, required=True, help='Something like cache_after_cuts_SR1_Rn220_21-11-2019.hdf. This should depend on your pattern. Make wise choice here.')
    parser.add_argument('--target_folder', type=str, default="/home/twolf/scratch-midway2", help="Folder in which the chunks of hdf-files are and the target_file will be written to.")
    parser.add_argument('--apply', action='store_true', help='Apply the merge.')

    args = vars(parser.parse_args())

    return args

def main(arg1):

    args = parse_args(arg1)
    path = args["target_folder"]
    target_file = args["target_file"]
    pattern = args["pattern"]

    if not os.path.exists(path):
        os.mkdir(path)

    #  pattern = "Part*_of_400_cache_before_cuts_SR1_Rn220_21-11-2019.hdf"

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



    if args["apply"]:
        l_dfs = []
        for this_file in filelist:
            print("Reading", this_file)
            df = dd.read_hdf(this_file, key='df_raw')
            df = df.query(combined_cut)
            l_dfs.append(df)
        print("Combining dask-dataframes")
        master_df = dd.concat(l_dfs)
        filename = os.path.join(path, target_file)
        master_df.to_hdf(filename, key='df_raw', format='table')
    else:
        print("DRY-RUN:")
        print("Gotten files:", filelist)
        print("Files in total:", len(filelist))

if __name__ == "__main__":
    main(sys.argv[1:])

