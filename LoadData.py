import datetime
import numpy as np
print("Starting now: {}".format(datetime.datetime.now()))
import os
import time as t

from pax import configuration
import tempfile
import subprocess
import shlex
import lax
import hax
import sys
import argparse
import yaml
import pandas as pd
import pickle
import dask.dataframe as dd

def check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue

def parse_args(args):
    parser = argparse.ArgumentParser(description='Load data from processed minitrees.')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--merge', action='store_true')
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--preliminary', action='store_true')
    parser.add_argument('--splits', type=check_positive, default=1)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--qos', type=str, default='xenon1t')
    parser.add_argument('--partition', type=str, default='xenon1t')
    parser.add_argument('--part_id', type=int, default=-1)
    parser.add_argument('--runs', nargs='*', default=None, required=False)

    args = vars(parser.parse_args())

    if args["debug"]:
      print("DEBUG turned on. Ignoring splits. Setting splits=1.")
      args["splits"] = 1

    with open(args['config'], 'r') as stream:
        try:
            parsed_config = (yaml.safe_load(stream))
        except yaml.YAMLError as exc:
            print(exc)
            raise SystemExit
    return args, parsed_config


def GetUniqueTagsOfDataset(dataset):
    return dataset.unique().tolist()


def ConfigurePax(parsed_args, parsed_config):
    pax_config = configuration.load_configuration('XENON1T')
    n_channels = pax_config['DEFAULT']['n_channels']
    pmts = pax_config['DEFAULT']['pmts']
    tpc_height = pax_config['DEFAULT']['tpc_length']
    tpc_radius = pax_config['DEFAULT']['tpc_radius']
    gains = pax_config['DEFAULT']['gains']

    pax_version = parsed_config["pax_settings"]["pax_version"]
    minitree_paths = ['/home/twolf/scratch-midway2', '/dali/lgrandi/xenon1t/minitrees/pax_v'+pax_version, '/project2/lgrandi/xenon1t/minitrees/pax_v'+pax_version, '/project/lgrandi/xenon1t/minitrees/pax_v'+pax_version]
    print(minitree_paths)

    hax.__version__
    hax.init(experiment='XENON1T',
             pax_version_policy= pax_version,
             raw_data_access_mode = 'local',
             raw_data_local_path = ['project/lgrandi/xenon1t'],
             main_data_paths=['/dali/lgrandi/xenon1t/processed/pax_v' + pax_version],
             minitree_paths=minitree_paths,
             make_minitrees = False,
             log_level='DEBUG',
             #  pax_version_policy='loose'
            )
    return hax

def SelectDataAccordingToType(parsed_config, pax_settings, dsets, datasets):
    dsets_type = []
    for item in parsed_config["data_type"]:
        dsets_type.append(dsets[(datasets.source__type == item)])
    dsets_type = pd.concat(dsets_type)
    print('Total {} datasets: We are left with {} datasets'.format(parsed_config["data_type"], len(dsets_type)))

    # select the latest versions
    dsets_type = dsets_type[(dsets_type.pax_version == pax_settings["pax_version"])] #
    print('Pax version: We are left with {} {} datasets'.format(len(dsets_type), parsed_config["data_type"]))

    # Select tags
    dsets_type = hax.runs.tags_selection(dsets_type, include=pax_settings['tags_to_include'],
          exclude=pax_settings['tags_to_exclude'])

    print('Remove bad tags: We are left with {} {} datasets'.format(len(dsets_type), parsed_config["data_type"]))

    # Select with a processed location
    dsets_type = dsets_type[(dsets_type.location != '')]
    print('Have location: We are left with {} {} datasets'.format(len(dsets_type), parsed_config["data_type"]))
    return dsets_type

def ConfigureLax():
    lax_version = lax.__version__

def SubmitToCluster(splitted_arrays, pax_settings, parsed_config, parsed_args):
    text = """#!/bin/bash
#SBATCH --job-name=part{part_id}_of_{splits}_{config}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8000
#SBATCH --output={logfile}
#SBATCH --error={logfile}
#SBATCH --account=pi-lgrandi
#SBATCH --qos={qos}
#SBATCH --partition={partition}
#SBATCH --mail-user=twolf@mpi-hd.mpg.de
#SBATCH --mail-type=NONE
export PATH=/project/lgrandi/anaconda3/bin:$PATH
echo PATH=$PATH

cp CustomCutsToApply.py ${{TMPDIR}}
cp LoadData.py ${{TMPDIR}}
cp {config} ${{TMPDIR}}
cd ${{TMPDIR}}
ls -ltrha
rm -f pax_event_class*
#  source activate pax_head
source activate pax_v6.10.1
python LoadData.py --config {config_in_tmp} --load --runs {runs} --part_id {part_id} --splits {splits}
echo "python LoadData.py --config {config_in_tmp} --load --runs {runs} --part_id {part_id} --splits {splits}"
ls -ltrha
mv *.pkl {outdir}

    """
    for part_id, runs in enumerate(splitted_arrays):
        logfile = GetFilename(parsed_config, part_id, parsed_args["splits"])
        logfile = logfile.replace(".hdf", ".log")
        config_in_tmp = os.path.basename(parsed_args["config"])
        runs_string = "'" +  "' '".join(runs) + "'"
        y = text.format(config=parsed_args["config"], runs=runs_string, part_id=part_id, splits=parsed_args['splits'], qos=parsed_args['qos'], partition=parsed_args['partition'], logfile=logfile, outdir=parsed_config["outdir"],
            config_in_tmp=config_in_tmp)
        submit_job(y)


def submit_job(sbatch_script):
    _, file = tempfile.mkstemp(suffix='.sbatch')
    with open(file, 'w') as f:
        f.write(sbatch_script)

    command = "sbatch %s" % file
    print("Executing: %s" % command )
    subprocess.Popen(shlex.split(command)).communicate()
    os.remove(file)


def PicklePerRuns(part_id, length, part_run_names, pax_settings, parsed_config, parsed_args):
    # Load the minitrees
    minitrees_to_load = pax_settings["minitrees_to_load"]
    preselection = pax_settings["preselection"]
    df = hax.minitrees.load(part_run_names, minitrees_to_load, preselection=preselection, num_workers=4)

    cut_names = []
    for cut in parsed_config['official_cuts_to_apply']:
      exec("global lichens; lichens = " + cut)
      # lichens = lax.lichens.sciencerun1.LowEnergyBackground()
      cut_names += lichens.get_cut_names()

      #Now run the lichens over the data we already loaded and get the booleans
      df = lichens.process(df)

    for cut in parsed_config["custom_cuts_to_apply"]:
      import CustomCutsToApply
      print("Executing cut: ", cut)
      exec(cut)


    print("Gotten cuts:", cut_names)
    #file_name = 'cache_before_cuts_SR2_Bkg_' + t.strftime("%d-%m-%Y") + '.pkl'

    filename = GetFilename(parsed_config, part_id, length)

    if parsed_args["debug"]:
      file_split = os.path.split(filename)
      filename = os.path.join(file_split[0], "DEBUG_" + file_split[1])
    print(filename)

    df.to_hdf(filename, key='df_raw', format='table')  # where to save it, usually as a .pkl

def GetFilename(parsed_config, part_id, length):
    filename = os.path.join(parsed_config['outdir'], "Part{part}_of_{length}_{filename_base}_{data_type}_{time}.hdf".format(part=part_id,
                                                                   filename_base=parsed_config["filename_base"],
                                                                   data_type=parsed_config["data_type_for_filename"],
                                                                   time=t.strftime("%d-%m-%Y"),
                                                                   length=length))
    return filename

def MergeParts(parsed_args, parsed_config):
    if parsed_args["splits"] == 1:
        print("No merge needed!")
    else:
        print("Starting merge")
        l_files = []
        fail_to_merge = False
        for part_id in range(0, parsed_args["splits"]):
            filename = GetFilename(parsed_config, part_id, parsed_args["splits"])

            if not os.path.exists(filename):
              fail_to_merge = True
              print(filename, "does not exist. Cannot proceed to merge.")
            l_files.append(filename)

        if fail_to_merge:
            raise SystemExit

        l_data = [pd.read_pickle(this_file) for this_file in l_files]

        master_df = pd.concat(l_data)
        file_to_write = os.path.join(parsed_config["outdir"], "{filename_base}_{data_type}_{time}.pkl".format(filename_base=parsed_config["filename_base"],
                                                                   data_type=parsed_config["data_type_for_filename"],
                                                                   time=t.strftime("%d-%m-%Y")))
        master_df.to_pickle(file_to_write)


def main(arg1):
    parsed_args, parsed_config = parse_args(arg1)
    pax_settings = parsed_config["pax_settings"]

    ConfigurePax(parsed_args, parsed_config)

    if parsed_args['runs'] is not None:
        PicklePerRuns(parsed_args["part_id"], parsed_args["splits"], parsed_args["runs"], pax_settings, parsed_config, parsed_args)
        raise SystemExit

    datasets = hax.runs.datasets # this variable holds all dataset info

    # copy and set data
    dsets = datasets
    dsets['start_date'] = dsets.start.dt.date

    # select background data only
    dsets_bkg = dsets[(datasets.source__type == 'none') ]
    dsets_bkg = hax.runs.tags_selection(dsets_bkg, include=pax_settings['tags_to_include'],
          exclude=pax_settings['tags_to_exclude'])
    dsets_bkg = dsets_bkg[(dsets_bkg.location != '')]

    lifetime = (dsets_bkg["end"] - dsets_bkg["start"]).sum()
    lifetime_sec = lifetime.total_seconds()

    print("Total lifetime: %0.2f days" % float(lifetime_sec / 3600 / 24))
    print('%i datasets are used in this analysis.' % len(dsets_bkg.number))

    dsets_type = SelectDataAccordingToType(parsed_config, pax_settings, dsets, datasets)
    print("Tags of selected data-sets are: ", dsets_type.tags.unique())


    if parsed_args['debug']:
      run_names = dsets_type["name"].tolist()[:2]
    else:
      run_names = dsets_type["name"].tolist()

    minitrees_to_load = pax_settings['minitrees_to_load']
    preselection = pax_settings["preselection"]
    splitted_arrays = np.array_split(run_names, parsed_args["splits"])
    if parsed_args["load"]:
      print("Loading minitrees: {}".format(datetime.datetime.now()))
      if not parsed_args["submit"]:
        for part_idx, split in enumerate(splitted_arrays):
          PicklePerRuns(part_idx, len(splitted_arrays), split, pax_settings, parsed_config, parsed_args)
      else:
          SubmitToCluster(splitted_arrays, pax_settings, parsed_config, parsed_args)

    else:
      print("Loading minitrees not required.")

    if parsed_args["merge"]:
      MergeParts(parsed_args, parsed_config)


if __name__ == "__main__":
      main(sys.argv[:1])


#  from runDB import get_collection
#
#  collection = get_collection()
#  query = {'data': {'$elemMatch': {'host': 'midway-login1', 'status': 'transferred',
#                                       'pax_version': pax_version
#                                      }
#                               }
#                        }
#  cursor = collection.find(query, {‘number’: 1})
#  runlist = [r[‘number’] for r in cursor]

