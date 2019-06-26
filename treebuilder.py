import sys
import hax
#sys.path.append('/home/berget2/Documents/XeAnalysisScripts/Kr83m_1T/helpers/')
#from DoubleScatter import *

run = str(sys.argv[1])

pax_version = '6.10.1'

hax.__version__

hax.init(experiment='XENON1T',
         pax_version_policy= pax_version,
         raw_data_access_mode = 'local',
         raw_data_local_path = ['project/lgrandi/xenon1t'],
         main_data_paths=['/dali/lgrandi/xenon1t/processed/pax_v' + pax_version],
         minitree_paths = ['.', '/dali/lgrandi/xenon1t/minitrees/pax_v'+pax_version,'/project2/lgrandi/xenon1t/minitrees/pax_v'+pax_version],
         make_minitrees = True
)

minitrees_to_load = [
                     'Corrections',
                     'Basics',
                     'Fundamentals',
                     #'Extended', 'LargestPeakProperties',
                     #  'TotalProperties', 'Proximity', 'FlashIdentification', 'TailCut', 'PatternReconstruction',
                     "CorrectedDoubleS1Scatter"
                     ]

hax.minitrees.load(run, minitrees_to_load, preselection = ['cs1<500',"s2>200",'cs1 > 0','cs2 > 0'])
