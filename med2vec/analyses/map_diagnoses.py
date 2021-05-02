import pickle
import argparse

# creates and pickles two dictionaries:
# seqs_to_icd9.dict
# labels_to_icd9.dict
#
# these can later be unpickled to "translate" the Med2Vec code into a MIMIC-III diagnosis code
# that can then be checked for its grouping into the 25 care conditions  that are part
# of the Benchmark phenotyping task

parser = argparse.ArgumentParser(description='Create diagnosis map from Med2Vec-style codes '
                                             'to codes in  MIMIC-III DIAGNOSES.csv')
parser.add_argument('mimic3_path', type=str, help='Directory containing MIMIC-III CSV files.')
args, _ = parser.parse_known_args()

diagnosisFile = args.mimic3_path


def convert_to_icd9(dxStr):
    if dxStr.startswith('E'):
        if len(dxStr) > 4:
            return dxStr[:4] + '.' + dxStr[4:]
        else:
            return dxStr
    else:
        if len(dxStr) > 3:
            return dxStr[:3] + '.' + dxStr[3:]
        else:
            return dxStr


def convert_to_3digit_icd9(dxStr):
    if dxStr.startswith('E'):
        if len(dxStr) > 4:
            return dxStr[:4]
        else:
            return dxStr
    else:
        if len(dxStr) > 3:
            return dxStr[:3]
        else:
            return dxStr


seqs_to_icd9 = {}
labels_to_icd9 = {}

infd = open(diagnosisFile, 'r')
infd.readline()
for line in infd:
    tokens = line.strip().split(',')
    dxCode = tokens[4][1:-1]
    dxStr = 'D_' + convert_to_icd9(tokens[4][1:-1])
    dxStr_3digit = 'D_' + convert_to_3digit_icd9(tokens[4][1:-1])

    if dxStr not in seqs_to_icd9:
        seqs_to_icd9[dxStr] = dxCode

    if dxStr_3digit not in labels_to_icd9:
        labels_to_icd9[dxStr_3digit] = [dxCode]
    else:
        if dxCode not in labels_to_icd9[dxStr_3digit]:
            labels_to_icd9[dxStr_3digit].append(dxCode)

with open('../resources/seqs_to_icd.dict', 'wb') as f1:
    pickle.dump(seqs_to_icd9, f1)

with open('../resources/labels_to_icd.dict', 'wb') as f1:
    pickle.dump(labels_to_icd9, f1)
