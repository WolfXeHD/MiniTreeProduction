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
