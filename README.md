# MiniTreeProduction
This repo serves as skimming of minitrees to produce .pkl files.

## Requirements on Midway
1. `export PATH="/project/lgrandi/anaconda3/bin:$PATH"`
2. `git clone https://github.com/WolfXeHD/MiniTreeProduction.git`
3. `module avail git`
4. `source activate pax_head`

### Usage to produce minitrees
1. Customize the .yaml file for your desires
2. `python BatchBuilder.py --config YOUR_YAML.yaml` submits the jobs per run to the cluster. Consider testing with the `--debug`-flag. The script calls on the cluster `python treebuilder.py {run}`. Consider checking this locally without submission.

### Usage to produce pickle-file
0. All minitrees should be present in some hax-version
1. `python LoadData.py --conifg YOUR_YAML.yaml` - consider testing it with the `--debug`-flag.
2. The pkl-file should be written in the current directory which can be extracted with the data-analysis tools of your choice.
