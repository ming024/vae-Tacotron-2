import argparse
import os
import shutil
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
from parselmouth.praat import call, run_file

def run(args):

    print('Loading metadata...')
    with open(args.meta, encoding='utf-8') as f:
        filenames = [line.strip().split('|')[0][6:-4] for line in f]
    
    print('computing audio statistics...')
    
    speech_rates = {}
    F0_means = {}
    F0_stds = {}
    logdir = './.praat_temp_'
    os.makedirs(logdir, exist_ok = False)
    for filename in tqdm(filenames):
        if args.dataset == 'Blizzard-2012':
            bookname = filename[:filename.find('_')] 
            filepath = os.path.join(args.input_dir, bookname, 'wav', filename[filename.find('_') + 1:] +'.wav')
        else:
            filepath = os.path.join(args.input_dir, filename + '.wav')

        source_run='audio_info.praat'
        
        try:
            objects = run_file(source_run, -20, 2, 0.3, "yes", filepath, logdir, 80, 400, 0.01, capture_output=True)
        
            z1=str( objects[1])
            z2=z1.strip().split()
            z3=np.array(z2)
            z4=np.array(z3)[np.newaxis]
            z5=z4.T

            syllable = int(z5[0, :])
            pause = int(z5[1, :])
            speaking_duration = float(z5[4, :])
            original_duration = float(z5[5, :])
            F0_mean = float(z5[7, :])
            F0_std = float(z5[8, :])
            F0_median = float(z5[9, :])
            F0_min = float(z5[10, :])
            F0_max = float(z5[11, :])

            speech_rate = syllable / original_duration
            
            speech_rates[filename] = speech_rate
            F0_means[filename] = F0_mean
            F0_stds[filename] = F0_std

        except:
            continue

    shutil.rmtree(logdir, ignore_errors = True)
    with open(os.path.join(args.pkl_dir, 'speech_rates.pkl'), 'wb') as f:
            pickle.dump(speech_rates, f)
    with open(os.path.join(args.pkl_dir, 'F0_means.pkl'), 'wb') as f:
            pickle.dump(F0_means, f)
    with open(os.path.join(args.pkl_dir, 'F0_stds.pkl'), 'wb') as f:
            pickle.dump(F0_stds, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Blizzard-2012', help='Blizzard-2012. Blizzard-2013, LJSpeech, or VCTK')
    parser.add_argument('--meta', default='/groups/ming/tacotron2/Blizzard-2012/data/train.txt', help='path to the metadata file.')
    parser.add_argument('--pkl_dir', default = '/groups/ming/tacotron2/Blizzard-2012/tacotron_output/', help='path to save/load the pickle files.')
    parser.add_argument('--input_dir', default='/groups/ming/data/Blizzard-2012', help='folder that contain the input wav files.')

    args = parser.parse_args()
    
    run(args)

if __name__ == '__main__':
    main()
