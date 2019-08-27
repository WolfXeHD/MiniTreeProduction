# 2016-09, Script from Chris and Sander

# log files will be created in the folder where you run this script
# For every job you will get an email, make sure this won't be blocked as it will look like spam if you get 500+ emails
# You need write permission for making the processing_dir folder

x = """#!/bin/bash
#SBATCH --job-name={run}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8000
#SBATCH --output=minitree_%J.log
#SBATCH --error=minitree_%J.log
#SBATCH --account=pi-lgrandi
#SBATCH --qos={qos}
#SBATCH --partition={partition}
#SBATCH --mail-user=twolf@mpi-hd.mpg.de
#SBATCH --mail-type=NONE
export PATH=/project/lgrandi/anaconda3/bin:$PATH
export PROCESSING_DIR=/home/twolf/scratch-midway2/production_{run}
echo PATH=$PATH

mkdir -p ${{PROCESSING_DIR}}

cp treebuilder.py ${{PROCESSING_DIR}}
cd ${{PROCESSING_DIR}}
ls -ltrha
rm -f pax_event_class*
source activate pax_head
python treebuilder.py {run}
ls -ltrha
mv *root /home/twolf/scratch-midway2
"""

import LoadData
import yaml
import sys
# Use submit procedure from CAX
from cax.qsub import submit_job
pax_version = '6.10.1'

# setup hax for Midway JupyterHub
import hax
import argparse

def parse_args(args):
    parser = argparse.ArgumentParser(description='Load data from processed minitrees.')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--qos', type=str, default='xenon1t')
    parser.add_argument('--partition', type=str, default='xenon1t')

    args = vars(parser.parse_args())

    with open(args['config'], 'r') as stream:
        try:
            parsed_config = (yaml.safe_load(stream))
        except yaml.YAMLError as exc:
            print(exc)
            raise SystemExit
    return args, parsed_config




hax.__version__
hax.init(experiment='XENON1T',
         pax_version_policy=pax_version,
         raw_data_access_mode='local',
         raw_data_local_path=['project/lgrandi/xenon1t'],
         main_data_paths=['/dali/lgrandi/xenon1t/processed/pax_v' + pax_version],
         minitree_paths = ['.', '/dali/lgrandi/xenon1t/minitrees/pax_v'+pax_version,'/project2/lgrandi/xenon1t/minitrees/pax_v'+pax_version],
         make_minitrees = True,
         # pax_version_policy='loose'
        )

datasets = hax.runs.datasets # this variable holds all dataset info
dsets = datasets
dsets['start_date'] = dsets.start.dt.date
parsed_args, parsed_config = parse_args(sys.argv[:1])
pax_settings = parsed_config["pax_settings"]
dsets_type = LoadData.SelectDataAccordingToType(parsed_config, pax_settings, dsets, datasets)

runs_to_submit = dsets_type["name"]
if parsed_args["debug"]:
  runs_to_submit = runs_to_submit[-2:]

# For every run, make and submit the script
for run in runs_to_submit:
    y = x.format(run=run, qos=parsed_args['qos'], partition=parsed_args['partition'])
    submit_job(y)

# Check your jobs with: 'qstat -u <username>'
# Check number of submitted jobs with 'qstat -u <username> | wc -l' (is off by +2 btw)
