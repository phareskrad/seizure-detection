#add dummy scores to missing test files

import os
import json
import pandas as pd
import numpy as np


with open('settings.json') as f:
    settings = json.load(f)
root = os.path.dirname(os.getcwd())
submission_dir = os.path.join(root, str(settings['submission-dir']))
data_dir = os.path.join(root, str(settings['competition-data-dir']))
file_names = []
for target in ['1', '2', '3']:
    i = 0
    while True:
        i += 1
        filename = 'new_%s_%d.mat' % (target, i)
        if os.path.exists(data_dir + '/' + filename):
            file_names.append(filename)
        else:
            break

dummy = pd.DataFrame(file_names, columns=['File'])
dummy['Class'] = np.repeat(0, len(file_names))

submission_file = 'submission_submission1478979774103-RFe300s1bTruej5r0_time-correlation-r400-us-noeig.csv'
submission = pd.read_csv(submission_dir + '/' + submission_file)

def strip(text):
    try:
        return text.strip()
    except AttributeError:
        return text


revise_file = 'S1_testing_data_with_minimal_dropout_overlaps.csv'
revise = pd.read_csv(submission_dir + '/' + revise_file, sep=r',', names=['File'], converters={'File': strip})
revise = revise.drop_duplicates()



result = pd.merge(dummy[['File']], submission, how='left', on='File')
result = result.fillna(0)
#result.ix[result['File'].isin(revise['File']), 'Class']=1
result.to_csv(submission_dir + '/' + submission_file,index=False)