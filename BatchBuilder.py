# 2016-09, Script from Chris and Sander

# log files will be created in the folder where you run this script
# For every job you will get an email, make sure this won't be blocked as it will look like spam if you get 500+ emails
# You need write permission for making the processing_dir folder

x = """#!/bin/bash
#SBATCH --job-name={run}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8000
#SBATCH --mail-type=ALL
#SBATCH --output=minitree_%J.log
#SBATCH --error=minitree_%J.log
#SBATCH --account=pi-lgrandi
#SBATCH --qos=xenon1t
#SBATCH --partition=xenon1t
#SBATCH --mail-user=twolf@mpi-hd.mpg.de
#SBATCH --mail-type=NONE
export PATH=/project/lgrandi/anaconda3/bin:$PATH
#  export PROCESSING_DIR=/home/twolf/scratch-midway2/production_{run}
echo PATH=$PATH

cp treebuilder.py ${{TMPDIR}}
cd ${{TMPDIR}}
ls -ltrha
rm -f pax_event_class*
source activate pax_head
python treebuilder.py {run}
ls -ltrha
mv *root /home/twolf/scratch-midway2
"""

import LoadData
import sys
# Use submit procedure from CAX
from cax.qsub import submit_job
pax_version = '6.10.1'

# setup hax for Midway JupyterHub
import hax
hax.__version__
hax.init(experiment='XENON1T',
         pax_version_policy=pax_version,
         raw_data_access_mode='local',
         raw_data_local_path=['project/lgrandi/xenon1t'],
         main_data_paths=['/dali/lgrandi/xenon1t/processed/pax_v' + pax_version],
         #minitree_paths = ['/home/hasterok/minitrees/pax_v'+pax_version,'/project/lgrandi/xenon1t/minitrees/pax_v'+pax_version,'/project2/lgrandi/xenon1t/minitrees/pax_v'+pax_version],
         minitree_paths = ['.', '/dali/lgrandi/xenon1t/minitrees/pax_v'+pax_version,'/project2/lgrandi/xenon1t/minitrees/pax_v'+pax_version],
         make_minitrees = True,
        #pax_version_policy='loose'
        )

datasets = hax.runs.datasets # this variable holds all dataset info
dsets = datasets
dsets['start_date'] = dsets.start.dt.date
parsed_args, parsed_config = LoadData.parse_args(sys.argv[:1])
pax_settings = parsed_config["pax_settings"]
dsets_type = LoadData.SelectDataAccordingToType(parsed_config, pax_settings, dsets, datasets)

runs_to_submit = dsets_type["name"]
if parsed_args["debug"]:
  runs_to_submit = runs_to_submit[-2:]

# For every run, make and submit the script
for run in runs_to_submit:
    y = x.format(run=run)
    submit_job(y)

# Check your jobs with: 'qstat -u <username>'
# Check number of submitted jobs with 'qstat -u <username> | wc -l' (is off by +2 btw)
