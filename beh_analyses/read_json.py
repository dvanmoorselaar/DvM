import json
import glob
import pandas as pd
import numpy as np
 
from IPython import embed

if hasattr(json.decoder, 'JSONDecodeError'):
    jsonerror = json.decoder.JSONDecodeError
else:
    jsonerror = ValueError

# get all raw result files
#files = glob.glob('/Users/dvm/Documents/projects/covert_sequence/raw/*.txt')
#files = glob.glob('/Users/dvm/Documents/projects/d_prime/raw/*.txt')
#files = glob.glob('/Users/dvm/Documents/projects/dist_onset/RT/raw/*.txt')
#files = glob.glob('/Users/dvm/Documents/projects/dist_onset/RT_exp2/raw/*.txt')
#files = glob.glob('/Users/dvm/Documents/projects/dist_onset/RT_exp3/raw/*.txt')
#files = glob.glob('/Users/dvm/Documents/projects/dist_onset/RT_exp4/raw/*.txt')
#files = glob.glob('/Users/dvm/Documents/projects/present_absent/target_SL/exp_1/raw/*.txt')
#files = glob.glob('/Users/dvm/Documents/projects/present_absent/target_SL/exp_2/raw/*.txt')
#files = glob.glob('/Users/dvm/Documents/projects/present_absent/target_SL/exp_3/raw/*.txt')
#files = glob.glob('/Users/dvm/Documents/projects/present_absent/dist_SL_dprime/raw/*.txt')
#files = glob.glob('/Users/dvm/Documents/projects/spatial_transfer/object/raw/*.txt')
#files = glob.glob('/Users/dvm/Documents/projects/spatial_transfer/control/raw/*.txt')
#files = glob.glob('/Users/dvm/Documents/projects/spatial_transfer/object_barble_cross/raw/*.txt')
files = glob.glob('/Users/dvm/Documents/projects/probe/exp_5/raw/*.txt')

# set general parameters
control_sj = True
column_idx = 'nr_trials'
sjs = []
comb = []


for f, file in enumerate(files):
    print('start reading file {} out of {} files'.format(f+1, len(files)))
    dfs = []
    partial = []

    with open(file, "r") as fd: 
        # loop over each logged datafile
        for line in fd: 
            # strip the lines so that they are valid json
            line = line.strip()
            # strip unwanted characters
            if line.startswith('[{'): 
                line = line[1:]     
            if not line.endswith('}') and '}' in line: 
                idx = len(line) - line[::-1].index('}') 
                line = line[:idx]
            try: 
                d = json.loads(line) 
            except: 
                continue

            # data from succesfully finished experiments
            if 'data' in d:
                if d['data'] != []:
                    df = pd.DataFrame(d['data'])
                    while control_sj and df.loc[0,'subject_nr'] in sjs:
                        df.loc[:,'subject_nr'] = '{}#'.format(df.loc[0,'subject_nr'])
                    dfs.append(df)
                    sjs.append(df.loc[0,'subject_nr'])
            # data from incomplete experimemts
            elif d != []:
                # store all incomplete files as dictionaries
                partial.append(d)

    # print info
    print('There are {} complete data files, with the following IDs: \n{}'.format(len(sjs), sjs))

    # check whether dataset contains partial log files
    if partial != []:
        print('Data set contains incomplete log files')
        partial = pd.DataFrame(partial, dtype=object).fillna('')
        if control_sj:
            prev = 0
            for idx, sj in partial['subject_nr'].iteritems():
                while control_sj and partial.loc[idx,'subject_nr'] in sjs:
                    partial.loc[idx,'subject_nr'] = '{}#'.format(partial.loc[idx,'subject_nr'])
                if partial.loc[idx, column_idx] < prev: # start new subject
                    sjs.append(partial.loc[idx-1, 'subject_nr'])
                prev = partial.loc[idx, column_idx]

        sj_info = np.unique(partial.subject_nr)
        print('There are {} incomplete data files, with the following IDs: \n{}'.format(sj_info.size, list(sj_info)))

    # create final dataframe
    if isinstance(partial, pd.DataFrame) and dfs != []:
        df = pd.concat([pd.concat(dfs, sort = True), partial], sort = True, ignore_index = True)
    elif isinstance(partial, pd.DataFrame):
        df = partial
    elif dfs != []:
        df = pd.concat(dfs, sort = True, ignore_index = True)
    else:
        continue

    comb.append(df)

if len(comb)>1:
    df =  pd.concat(comb, sort = True, ignore_index = True)  
elif len(comb) == 1:
    df = comb[0]
else:
    print('no data file found')

# save datafile 
#df.to_csv('/Users/dvm/Documents/projects/covert_sequence/raw/raw_data.csv')
#df.to_csv('/Users/dvm/Documents/projects/d_prime/raw/raw_data.csv')
#df.to_csv('/Users/dvm/Documents/projects/dist_onset/RT/raw/raw_data.csv')
#df.to_csv('/Users/dvm/Documents/projects/dist_onset/RT_exp2/raw/raw_data.csv')
#df.to_csv('/Users/dvm/Documents/projects/dist_onset/RT_exp3/raw/raw_data.csv')
#df.to_csv('/Users/dvm/Documents/projects/dist_onset/RT_exp4/raw/raw_data.csv')
#df.to_csv('/Users/dvm/Documents/projects/present_absent/target_SL/exp_1/raw/raw_data.csv')
#df.to_csv('/Users/dvm/Documents/projects/present_absent/target_SL/exp_2/raw/raw_data.csv')
#df.to_csv('/Users/dvm/Documents/projects/present_absent/target_SL/exp_3/raw/raw_data.csv')
#df.to_csv('/Users/dvm/Documents/projects/present_absent/dist_SL_dprime/raw/raw_data.csv') 
#df.to_csv('/Users/dvm/Documents/projects/present_absent/dist_SL_dprime/raw/raw_data.csv')
#df.to_csv('/Users/dvm/Documents/projects/spatial_transfer/object/raw/raw_data.csv')
#df.to_csv('/Users/dvm/Documents/projects/spatial_transfer/control/raw/raw_data.csv')
#df.to_csv('/Users/dvm/Documents/projects/spatial_transfer/object_barble_cross/raw/raw_data.csv')
df.to_csv('/Users/dvm/Documents/projects/probe/exp_5/raw/raw_data.csv')
         
