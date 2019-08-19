import datetime
import numpy as np
print("Starting now: {}".format(datetime.datetime.now()))
#  import ROOT
#  import numpy as np
#  import root_numpy
#  import math
#  import os
#  import pandas as pd
#  import matplotlib.pyplot as plt
#  import matplotlib

import time as t

from pax import configuration
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
  parser.add_argument('--splits', type=check_positive, default=1)
  parser.add_argument('--config', type=str, required=True)

  args = vars(parser.parse_args())

  with open(args['config'], 'r') as stream:
      try:
          parsed_config = (yaml.safe_load(stream))
      except yaml.YAMLError as exc:
          print(exc)
          raise SystemExit
  return args, parsed_config


def GetUniqueTagsOfDataset(dataset):
  return dataset.unique().tolist()


def ConfigurePax():
  """TODO: Docstring for ConfigurePax.
  :returns: TODO

  """
  pax_config = configuration.load_configuration('XENON1T')
  n_channels = pax_config['DEFAULT']['n_channels']
  pmts = pax_config['DEFAULT']['pmts']
  tpc_height = pax_config['DEFAULT']['tpc_length']
  tpc_radius = pax_config['DEFAULT']['tpc_radius']
  gains = pax_config['DEFAULT']['gains']

  pax_version = '6.10.1'
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
  dsets_type = hax.runs.tags_selection(dsets_type, include=['sciencerun2_preliminary'],
        exclude=pax_settings['tags_to_exclude'])

  print('Remove bad tags: We are left with {} {} datasets'.format(len(dsets_type), parsed_config["data_type"]))

  # Select with a processed location
  dsets_type = dsets_type[(dsets_type.location != '')]
  print('Have location: We are left with {} {} datasets'.format(len(dsets_type), parsed_config["data_type"]))
  return dsets_type


def ConfigureLax():
  lax_version = lax.__version__


def PicklePerRuns(part_id, length, part_run_names, pax_settings, parsed_config, parsed_args):
  # Load the minitrees
  minitrees_to_load = pax_settings["minitrees_to_load"]
  preselection = pax_settings["preselection"]
  df = hax.minitrees.load(part_run_names, minitrees_to_load, preselection=preselection, num_workers=4)


  # get all cience run 0 cuts
  sr1_cuts = lax.lichens.sciencerun1.LowEnergyBackground()
  cut_names = lax.lichens.sciencerun1.LowEnergyBackground().get_cut_names()
  print(cut_names)

  #Now run the lichens over the data we already loaded and get the booleans
  data = sr1_cuts.process(df)

  #file_name = 'cache_before_cuts_SR2_Bkg_' + t.strftime("%d-%m-%Y") + '.pkl'
  filename_base = parsed_config["filename_base"]

  if part_id != -1:
    filename = "Part{part}_of_{length}_{filename_base}_{data_type}_{time}.pkl".format(part=part_id,
                                                                     filename_base=parsed_config["filename_base"],
                                                                     data_type=parsed_config["data_type_for_filename"],
                                                                     time=t.strftime("%d-%m-%Y"),
                                                                     length=length)
  else:
    filename = "{filename_base}_{data_type}_{time}.pkl".format(filename_base=parsed_config["filename_base"],
                                                               data_type=parsed_config["data_type_for_filename"],
                                                               time=t.strftime("%d-%m-%Y"))
  if parsed_args["debug"]:
    filename = "DEBUG_" + filename
  print(filename)

  data.to_pickle(filename)  # where to save it, usually as a .pkl


def MergeParts(parsed_args, parsed_config):
  if parsed_args["splits"] == 1:
    print("No merge needed!")
  else:
    l_data = []
    for part_id in range(0, parsed_args["splits"]):
      file_to_open = "Part{part}_of_{length}_{filename_base}_{data_type}_{time}.pkl".format(part=part_id,
                                                                       filename_base=parsed_config["filename_base"],
                                                                       data_type=parsed_config["data_type_for_filename"],
                                                                       time=t.strftime("%d-%m-%Y"),
                                                                       length=parsed_args["splits"])


      l_data.append(pd.read_pickle(file_to_open))
    master_df = pd.concat(l_data)
    file_to_write = "{filename_base}_{data_type}_{time}.pkl".format(filename_base=parsed_config["filename_base"],
                                                               data_type=parsed_config["data_type_for_filename"],
                                                               time=t.strftime("%d-%m-%Y"))
    master_df.to_pickle(file_to_write)


def main(arg1):
  parsed_args, parsed_config = parse_args(arg1)
  pax_settings = parsed_config["pax_settings"]

  ConfigurePax()

  datasets = hax.runs.datasets # this variable holds all dataset info

  # copy and set data
  dsets = datasets
  dsets['start_date'] = dsets.start.dt.date

  # select background data only
  dsets_bkg = dsets[(datasets.source__type == 'none') ]
  dsets_bkg = hax.runs.tags_selection(dsets_bkg, include=['sciencerun2_preliminary'],
        exclude=pax_settings['tags_to_exclude'])
  dsets_bkg = dsets_bkg[(dsets_bkg.location != '')]

  lifetime = (dsets_bkg["end"] - dsets_bkg["start"]).sum()
  lifetime_sec = lifetime.total_seconds()

  print("Total lifetime: %0.2f days" % float(lifetime_sec / 3600 / 24))
  print('%i datasets are used in this analysis.' % len(dsets_bkg.number))

  dsets_type = SelectDataAccordingToType(parsed_config, pax_settings, dsets, datasets)

  if parsed_args['debug']:
    run_names = dsets_type["name"].tolist()[:10]
  else:
    run_names = dsets_type["name"].tolist()

  if parsed_args["load"]:
    print("Loading minitrees: {}".format(datetime.datetime.now()))
    if parsed_args["splits"] != 1:
      splitted_arrays = np.array_split(run_names, parsed_args["splits"])
      minitrees_to_load = pax_settings['minitrees_to_load']
      preselection = pax_settings["preselection"]

      for part_idx, split in enumerate(splitted_arrays):
        PicklePerRuns(part_idx, len(splitted_arrays), split, pax_settings, parsed_config, parsed_args)
    else:
        PicklePerRuns(-1, 1, run_names, pax_settings, parsed_config, parsed_args)
  else:
    print("Loading minitrees not required.")

  if parsed_args["merge"]:
    MergeParts(parsed_args, parsed_config)


if __name__ == "__main__":
      main(sys.argv[:1])
