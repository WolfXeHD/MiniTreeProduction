import numpy as np
import pickle


def classify(df):
      with open('/project2/lgrandi/zhut/s2_single_classifier_gmix_v6.10.0.pkl', 'rb') as f:
            gmix=pickle.load(f)
      df['CutS2SingleScatterHE'] = 0

      mask = df.eval('(largest_other_s2>0) & (s2>0) & (largest_other_s2_pattern_fit>0)')
      Y = np.concatenate([np.log10(df.loc[mask,['largest_other_s2', 'largest_other_s2_pattern_fit', 's2']]),
                      ],
                     axis=1)
      df.loc[mask,'CutS2SingleScatterHE'] = gmix.predict(Y)
      df.loc[:, 'CutS2SingleScatterHE'] = np.array(df.CutS2SingleScatterHE, bool)
      return df

def CutS2SingleScatterHE(df):
    df = classify(df)

def CutPosDiffHE(df):
    df['d_tf'] = np.sqrt((df['x_observed_nn_tf']-df['x_observed_tpf'])**2 + (df['y_observed_nn_tf']-df['y_observed_tpf'])**2)
    df['CutPosDiffHE'] = df.d_tf < 3569.674 * np.exp(-np.log10(df.s2)/0.369) + 1.582


# class S1AreaFractionTop_he(Lichen):
#
#     """Cut between in the [0.1 - 99.9] percentile of the population in the parameter space Z vs S1AFT
#     Note: https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenon1t:arianna:s1_aft_highenergy
#        Contact: arianna.rocchetti@physik.uni-freiburg.de
#        Cut defined above cs1>200.
#        Requires: PatternReconstruction Minitree.
#     """
#     version = 0
#
def f1(x):
    f1 = -315.6*x**2 + 445.6 * x - 153.7
    return f1
def f2(x):
    f2 = 188.7*x**3 - 1172*x**2 + 719.7* x - 119.2
    return f2


def S1AreaFractionTop_he(df):
    print("Warning: Cut defined above cs1>200. ")

    df['CutS1AreaFractionTopHE'] = 0

    df.loc[:, 'CutS1AreaFractionTopHE'] = np.array(df.CutS1AreaFractionTopHE, bool)

    df.CutS1AreaFractionTopHE[(df.z_3d_nn_tf > f1(df.s1_area_fraction_top)) & (df.z_3d_nn_tf < f2(df.s1_area_fraction_top) )] = True
    df.CutS1AreaFractionTopHE[(df.z_3d_nn_tf < f1(df.s1_area_fraction_top)) & (df.z_3d_nn_tf > f2(df.s1_area_fraction_top) )] = False
    df = df[(df.z_3d_nn_tf > f1(df.s1_area_fraction_top)) & (df.z_3d_nn_tf < f2(df.s1_area_fraction_top))]
    return df
