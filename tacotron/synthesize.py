import argparse
import os
import re
import time
from time import sleep

import tensorflow as tf
from hparams import hparams, hparams_debug_string
from infolog import log
from tacotron.synthesizer import Synthesizer
from tqdm import tqdm


def generate_fast(model, text):
	model.synthesize([text], None, None, None, None)


def run_live(args, checkpoint_path, hparams):
	#Log to Terminal without keeping any records in files
	log(hparams_debug_string())
	synth = Synthesizer()
	synth.load(checkpoint_path, hparams)

	#Generate fast greeting message
	greetings = 'Hello, Welcome to the Live testing tool. Please type a message and I will try to read it!'
	log(greetings)
	generate_fast(synth, greetings)

	#Interaction loop
	while True:
		try:
			text = input()
			generate_fast(synth, text)

		except KeyboardInterrupt:
			leave = 'Thank you for testing our features. see you soon.'
			log(leave)
			generate_fast(synth, leave)
			sleep(2)
			break

def run_eval(args, checkpoint_path, output_dir, hparams, sentences):
	eval_dir = os.path.join(output_dir, 'eval')
	log_dir = os.path.join(output_dir, 'logs-eval')
	if args.modify_vae_dim is not None:
		eval_dir += '-modify'
		log_dir += '-modify'
	modify_vae_dim = [int(dim) for dim in args.modify_vae_dim.split(',')] if args.modify_vae_dim else None
   
	if args.model == 'Tacotron-2':
		assert os.path.normpath(eval_dir) == os.path.normpath(args.mels_dir)

	#Create output path if it doesn't exist
	os.makedirs(eval_dir, exist_ok=True)
	os.makedirs(log_dir, exist_ok=True)

	log(hparams_debug_string())
	synth = Synthesizer()
	if args.reference_mel is not None and args.modify_vae_dim is None:
		synth.load(checkpoint_path, hparams, vae_code_mode='auto')
	elif args.reference_mel is not None and args.modify_vae_dim is not None:
		synth.load(checkpoint_path, hparams, vae_code_mode='modify')
	else:
		synth.load(checkpoint_path, hparams, vae_code_mode='feed')

	#Set inputs batch wise
	sentences = [sentences[i: i+hparams.tacotron_synthesis_batch_size] for i in range(0, len(sentences), hparams.tacotron_synthesis_batch_size)]

	log('Starting Synthesis')
	with open(os.path.join(eval_dir, 'map.txt'), 'w') as file:
		trange = tqdm(sentences)
		for i, texts in enumerate(trange):
			if args.modify_vae_dim is None:
				start = time.time()
				basenames = ['batch_{}_sentence_{}'.format(i, j) for j in range(len(texts))]
				if args.reference_mel is not None:
					mel_filenames = [args.reference_mel for j in range(len(texts))]
					mel_output_filenames, speaker_ids = synth.synthesize(texts, basenames, eval_dir, log_dir, mel_filenames)
				else:
					mel_output_filenames, speaker_ids = synth.synthesize(texts, basenames, eval_dir, log_dir, None)
				log_dir = None
				#save plots and wavs for the first batch only, for human inspection 
				
				for elems in zip(texts, mel_output_filenames, speaker_ids):
					file.write('|'.join([str(x) for x in elems]) + '\n')
			else:
				for dim in modify_vae_dim:
					for scale in [-2, -1, 0, 1, 2]:
						start = time.time()
						basenames = ['batch_{}_sentence_{}_dim_{}_mu+({}*sigma)'.format(i, j, dim, scale) for j in range(len(texts))]
						if args.reference_mel is not None:
							mel_filenames = [args.reference_mel for j in range(len(texts))]
							mel_output_filenames, speaker_ids = synth.synthesize(texts, basenames, eval_dir, log_dir, mel_filenames, dim, scale)
						else:
							mel_output_filenames, speaker_ids = synth.synthesize(texts, basenames, eval_dir, log_dir, None, dim, scale)

						trange.set_postfix({'modified_dim':dim, 'value':'mu+({}*sigma)'.format(scale)})
						trange.refresh()
						for elems in zip(texts, mel_output_filenames, speaker_ids):
							file.write('|'.join([str(x) for x in elems + (dim, scale)]) + '\n')
				log_dir = None
				#save plots and wavs for the first batch only, for human inspection 
	log('synthesized mel spectrograms at {}'.format(eval_dir))
	return eval_dir

def run_synthesis(args, checkpoint_path, output_dir, hparams):
	GTA = (args.GTA == 'True')
	if GTA:
		synth_dir = os.path.join(output_dir, 'gta')
		log_dir = os.path.join(output_dir, 'logs-gta')

		#Create output path if it doesn't exist
		os.makedirs(synth_dir, exist_ok=True)
		os.makedirs(log_dir, exist_ok=True)
	else:
		synth_dir = os.path.join(output_dir, 'natural')
		log_dir = os.path.join(output_dir, 'logs-natural')
		if args.modify_vae_dim is not None:
			synth_dir += '-modify'
			log_dir += '-modify'
		modify_vae_dim = [int(dim) for dim in args.modify_vae_dim.split(',')] if args.modify_vae_dim else None

		#Create output path if it doesn't exist
		os.makedirs(synth_dir, exist_ok=True)
		os.makedirs(log_dir, exist_ok=True)


	metadata_filename = os.path.join(args.input_dir, 'train.txt')
	log(hparams_debug_string())
	synth = Synthesizer()
	if GTA or args.modify_vae_dim is None:
		synth.load(checkpoint_path, hparams, gta=GTA, vae_code_mode='auto')
	else:
		synth.load(checkpoint_path, hparams, gta=GTA, vae_code_mode='modify')
	with open(metadata_filename, encoding='utf-8') as f:
		metadata = [line.strip().split('|') for line in f]
		frame_shift_ms = hparams.hop_size / hparams.sample_rate
		hours = sum([int(x[4]) for x in metadata]) * frame_shift_ms / (3600)
		log('Loaded metadata for {} examples ({:.2f} hours)'.format(len(metadata), hours))

	#Set inputs batch wise
	metadata = [metadata[i: i+hparams.tacotron_synthesis_batch_size] for i in range(0, len(metadata), hparams.tacotron_synthesis_batch_size)]

	log('Starting Synthesis')
	mel_dir = os.path.join(args.input_dir, 'mels')
	wav_dir = os.path.join(args.input_dir, 'audio')
	with open(os.path.join(synth_dir, 'map.txt'), 'w') as file:
		trange = tqdm(metadata)
		for i, meta in enumerate(trange):
			if GTA or args.modify_vae_dim is None:
				texts = [m[5] for m in meta]
				mel_filenames = [os.path.join(mel_dir, m[1]) for m in meta]
				wav_filenames = [os.path.join(wav_dir, m[0]) for m in meta]
				basenames = [os.path.basename(m).replace('.npy', '').replace('mel-', '') for m in mel_filenames]
				mel_output_filenames, speaker_ids = synth.synthesize(texts, basenames, synth_dir, log_dir, mel_filenames)
				log_dir = None
				#save plots and wavs for the first batch only, for human inspection 

				for elems in zip(wav_filenames, mel_filenames, mel_output_filenames, speaker_ids, texts):
					file.write('|'.join([str(x) for x in elems]) + '\n')
			else:
				for dim in modify_vae_dim:
					for scale in [-2, -1, 0, 1, 2]:
						texts = [m[5] for m in meta]
						mel_filenames = [os.path.join(mel_dir, m[1]) for m in meta]
						wav_filenames = [os.path.join(wav_dir, m[0]) for m in meta]
						basenames = [os.path.basename(m).replace('.npy', '').replace('mel-', '') + '-dim_{}_mu+({}*sigma)'.format(dim, scale) for m in mel_filenames]
						mel_output_filenames, speaker_ids = synth.synthesize(texts, basenames, synth_dir, log_dir, mel_filenames, dim, scale)
						trange.set_postfix({'modified_dim':dim, 'value':'mu+({}*sigma)'.format(scale)})
						trange.refresh()
						for elems in zip(wav_filenames, mel_filenames, mel_output_filenames, speaker_ids, texts):
							file.write('|'.join([str(x) for x in elems + (dim, scale)]) + '\n')
				break
				#synthesize spectrograms for the first batch only, for human inspection 
	log('synthesized mel spectrograms at {}'.format(synth_dir))
	return os.path.join(synth_dir, 'map.txt')

def tacotron_synthesize(args, hparams, checkpoint, sentences=None):
	output_dir = args.output_dir

	try:
		checkpoint_path = checkpoint
		log('loaded model at {}'.format(checkpoint))
	except:
		raise RuntimeError('Failed to load checkpoint at {}'.format(checkpoint))

	if hparams.tacotron_synthesis_batch_size < hparams.tacotron_num_gpus:
		raise ValueError('Defined synthesis batch size {} is smaller than minimum required {} (num_gpus)! Please verify your synthesis batch size choice.'.format(
			hparams.tacotron_synthesis_batch_size, hparams.tacotron_num_gpus))

	if hparams.tacotron_synthesis_batch_size % hparams.tacotron_num_gpus != 0:
		raise ValueError('Defined synthesis batch size {} is not a multiple of {} (num_gpus)! Please verify your synthesis batch size choice!'.format(
			hparams.tacotron_synthesis_batch_size, hparams.tacotron_num_gpus))

	if args.mode == 'eval':
		return run_eval(args, checkpoint_path, output_dir, hparams, sentences)
	elif args.mode == 'synthesis':
		return run_synthesis(args, checkpoint_path, output_dir, hparams)
	else:
		run_live(args, checkpoint_path, hparams)
