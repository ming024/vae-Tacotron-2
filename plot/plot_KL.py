import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def running_mean(arr, N = 100):
  tail = np.convolve(arr, np.ones((N))/N)[(N - 1):-(N)]
  head = np.array([np.mean(arr[:i + 1]) for i in range(N)])
  return np.append(head, tail)

vae_dim = 16
log_paths = ['/home/ming/Terminal_train_log',
  '/groups/ming/tacotron2/Blizzard-2012/logs-confidence=30/Terminal_train_log',
  '/groups/ming/tacotron2/Blizzard-2012/logs-confidence=-101/Terminal_train_log']
trace_names = ['confidence_threshold = 90', 'confidence_threshold = 30', 'confidence_threshold = -101']
name = 'KL_trace'
title = 'KL trace'

cmap = plt.cm.coolwarm
if len(log_paths) == 1:
  matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color = cmap(np.linspace(0, 1, vae_dim)))
else:
  matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color = cmap(np.linspace(0, 1, len(log_paths))))

# Read log files
KL_traces = [{} for _ in range(len(log_paths))]
step = None
mean = None
std = None
for i, (log_path, KL_trace) in enumerate(zip(log_paths, KL_traces)):
  with open(log_path) as log_file:
    for line in log_file:
      if line.find('Step ') != -1:
        step = int(line[line.find('Step') + 4:].strip(' ').split(' ')[0])
        KL_trace[step] = (0.5 * (std ** 2 + mean ** 2 - 1 - 2 * np.log(std)), mean, std)
      elif line.find('Mean:') != -1:
        mean = np.array([float(item) for item in re.split('[ \t\n\[\]]+', line[line.find('Mean:') + 5:] + log_file.readline()) if item])
      elif line.find('Std:') != -1:
        std = np.array([float(item) for item in re.split('[ \t\n\[\]]+', line[line.find('Std:') + 4:] + log_file.readline()) if item])
      else:
        continue
KL_traces = [sorted(list(KL_trace.items()), key = lambda tup: tup[0]) for KL_trace in KL_traces]

# Set axes, legends, and labels
fig, host = plt.subplots(figsize = (10,8))
host.set_xlabel("Steps")
host.set_ylabel("KL loss")
if len(log_paths) == 1:
  legend_elements = [matplotlib.lines.Line2D([0], [0], color = cmap(d / (vae_dim - 1)), linestyle = '-', label = 'dim_{}_KL_loss'.format(d)) for d in range(vae_dim)]
else:
  legend_elements = [matplotlib.lines.Line2D([0], [0], color = cmap(i / (len(log_paths) - 1)), linestyle = '-', label = trace_name) for i, trace_name in enumerate(trace_names)]
 
# Plot KL losses
if len(log_paths) == 1:
  steps, losses = zip(*KL_traces[0])
  KL_losses, means, stds = zip(*losses)

  steps = np.array(steps).astype(np.int32)
  dimensional_KL_losses = [running_mean(np.array([KL_loss[d] for KL_loss in KL_losses]).astype(np.float64), 500) for d in range(vae_dim)]
  for d, dimensional_KL_loss in enumerate(dimensional_KL_losses):
    host.plot(steps, dimensional_KL_loss, cmap(d / (vae_dim - 1)), linestyle = '-', linewidth = 2)
else:
  for i, KL_trace in enumerate(KL_traces):
    steps, losses = zip(*KL_trace)
    KL_losses, means, stds = zip(*losses)

    steps = np.array(steps).astype(np.int32)
    KL_losses = running_mean(np.array([np.sum(KL_loss) for KL_loss in KL_losses]).astype(np.float64), 500)
    host.plot(steps, KL_losses, cmap(i / (len(log_paths) - 1)), linestyle = '-', linewidth = 2)

# Plot
host.tick_params(axis = 'x')
host.tick_params(axis = 'y')
host.legend(handles = legend_elements)
plt.title(title)
plt.tight_layout()

plt.savefig(os.path.join(os.path.dirname(log_paths[0]), '{}.eps'.format(name)))
plt.savefig(os.path.join(os.path.dirname(log_paths[0]), '{}.png'.format(name)))
