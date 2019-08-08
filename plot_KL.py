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
log_path = '/groups/ming/tacotron2/Blizzard/logs-Tacotron-2/Terminal_train_log'
name = 'wo_freezing_with_teacher_forcing'
title = 'Without Freezing Tacotron Encoder and With Teacher Forcing'
teacher_forcing_init_ratio = 1.
teacher_forcing_final_ratio = 1.
teacher_forcing_start_decay = 10000
teacher_forcing_decay_steps = 40000

cmap = plt.cm.coolwarm
matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color = cmap(np.linspace(0, 1, vae_dim)))

# Read log file
KL_trace = {}
step = None
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
KL_trace = sorted(list(KL_trace.items()), key = lambda tup: tup[0])

# Set axes, legends, and labels
fig, host = plt.subplots(figsize = (10,8))
par1 = host.twinx()
host.set_xlabel("Steps")
host.set_ylabel("Dimensional KL loss")
par1.set_ylabel("Teacher Forcing Ratio")
legend_elements = [matplotlib.lines.Line2D([0], [0], color = cmap(d / (vae_dim - 1)), linestyle = '-', label = 'dim_{}_KL_loss'.format(d))
  for d in range(vae_dim)]
  
# Plot KL losses
steps, losses = zip(*KL_trace)
KL_losses, means, stds = zip(*losses)

steps = np.array(steps).astype(np.int32)
dimensional_KL_losses = [running_mean(np.array([KL_loss[d] for KL_loss in KL_losses]).astype(np.float64), 500) for d in range(vae_dim)]
for d, dimensional_KL_loss in enumerate(dimensional_KL_losses):
  host.plot(steps, dimensional_KL_loss, cmap(d / (vae_dim - 1)), linestyle = '-', linewidth = 2)

# Plot teacher forcing ratio
teacher_forcing_ratios = [teacher_forcing_init_ratio] * teacher_forcing_start_decay +\
  [0.5 * (teacher_forcing_init_ratio - teacher_forcing_final_ratio) * (1 + np.cos(np.pi * step / teacher_forcing_decay_steps)) +\
    teacher_forcing_final_ratio for step in range(teacher_forcing_decay_steps)] +\
  [teacher_forcing_final_ratio] * (len(steps) - teacher_forcing_start_decay - teacher_forcing_decay_steps)
p1, = par1.plot(steps, teacher_forcing_ratios[:len(steps)], 'b--', linewidth = 4)
legend_elements.append(matplotlib.lines.Line2D([0], [0], color = 'b', linestyle = '--', label = 'teacher_forcing_ratio'))

# Plot
host.tick_params(axis = 'x')
host.tick_params(axis = 'y')
par1.tick_params(axis = 'y')
host.set_ylim([0, 4])
par1.set_ylim([0, 1])
host.legend(handles = legend_elements)
plt.title(title)
plt.tight_layout()

plt.savefig(os.path.join(os.path.dirname(log_path), '{}.eps'.format(name)))
plt.savefig(os.path.join(os.path.dirname(log_path), '{}.png'.format(name)))