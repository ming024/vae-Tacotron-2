import argparse
import os
import pickle
import random
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def resampling(filenames, embeddings, audio_features, n_hubs, hub_size):
    # The distribution of the audio features, such as F0 and speech
    # rates, are highly non-uniform, implying that most of the data 
    # share some similar property. As a result, t-SNE cannot work well 
    # due to the severe central tendency.
    # In order to better visualize the audio segments with different
    # characteristics in the latent space, we first do re-sampling.
    # The audio samples are divided into [n_hubs] hubs along the 
    # feature axis within the 3-std-range, with the samples falling
    # out of this range discarded. Moreover, the size of each hub is 
    # fixed to ensure that the resulting data distribution is closed to
    # uniform. Unnecessary samples are discarded, and will not be
    # showed in the plots.

    mean = np.mean(list(audio_features.values()))
    std = np.std(list(audio_features.values()))
    hubs = [[] for _ in range(n_hubs)]

    random.shuffle(filenames)
    for filename in filenames:
        if filename not in audio_features:
            continue
        embedding = embeddings[filename]
        audio_feature = audio_features[filename]

        hub_index = (audio_feature - (mean - 3 * std)) / (6 * std) * n_hubs
        if hub_index < 0 or hub_index >= hub_size:
            continue
        else:
            hub_index = int(np.floor(hub_index))
            if len(hubs[hub_index]) < hub_size:
                hubs[hub_index].append((filename, embedding, audio_feature))
            else:
                continue

    hubs = [item for hub in hubs for item in hub]
    random.shuffle(hubs)
    return zip(*hubs)

def plot(embeddings, audio_features, figpath):
    print('Running t-SNE on {} smaples.'.format(len(embeddings)))
    tsne = TSNE(n_components = 2, verbose = 1, n_iter = 2000)
    projected = tsne.fit_transform(np.array(embeddings))
   
    print('Ploting t-SNE results...')
    scatter = plt.scatter(x = projected[:, 0], y = projected[:, 1], c = audio_features)
    plt.colorbar(scatter)
    plt.savefig(figpath)
    plt.clf()

def run(args):

    print('Loading metadata...')
    with open(args.meta, encoding='utf-8') as f:
        filenames = [line.strip().split('|')[0][6:-4] for line in f]
    
    print('Loading latent embeddings...')
    with open(os.path.join(args.pkl_dir, 'latent_embeddings.pkl'), 'rb') as f:
        latent_embeddings = pickle.load(f)

    print('Loading audio statistics...')
    with open(os.path.join(args.pkl_dir, 'speech_rates.pkl'), 'rb') as f:
        speech_rates = pickle.load(f)
    with open(os.path.join(args.pkl_dir, 'F0_means.pkl'), 'rb') as f:
        F0_means = pickle.load(f)
    with open(os.path.join(args.pkl_dir, 'F0_stds.pkl'), 'rb') as f:
        F0_stds = pickle.load(f)

    print('Total {} speech audio samples...'.format(len(filenames)))

    for features, feature_name in [(speech_rates, 'speech_rate'), (F0_means, 'f0_mean'), (F0_stds, 'f0_std')]:
        print('Resampling along the {} axis...'.format(feature_name))
        filenames_, latent_embeddings_, features = resampling(filenames, latent_embeddings, features, n_hubs = 50, hub_size = 50)
        plot(latent_embeddings_, features, os.path.join(args.output_dir, '{}.png'.format(feature_name)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta', default='/groups/ming/tacotron2/Blizzard-2012/data/train.txt', help='path to the metadata file.')
    parser.add_argument('--pkl_dir', default = '/groups/ming/tacotron2/Blizzard-2012/tacotron_output/', help='path to the folder that contains the pickled audio info/latent embedding files.')
    parser.add_argument('--output_dir', default = '/groups/ming/tacotron2/Blizzard-2012/tacotron_output/', help='folder to save the output plots.')
    args = parser.parse_args()
    
    run(args)

if __name__ == '__main__':
    main()
