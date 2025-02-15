#!/usr/bin/env python3
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

import model
import dataset

import imageio
import os
from typing import Tuple, List
import matplotlib.pyplot as plt
from matplotlib import collections as matcoll

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string("logs_dir", "logs", "Where the log file is saved.")
flags.DEFINE_string("logs_file", "logs.txt", "Where the logs are saved.")
flags.DEFINE_string("loss_binary", "./loss.py", "Binary to loss file")
flags.DEFINE_integer("plot_every_n_iterations", 10,
                     "How often to plot learning progress.")

dataset = dataset.MaskingDataset()
for filename in os.listdir("data"):
  if filename.startswith("masker") and filename.endswith(".txt"):
    dataset.read_data("data", filename)

masking_frequency = float(os.environ["MASK_FREQ"])
probe_level = int(os.environ["PROBE_LEVEL"])
masking_level = int(os.environ["MASKING_LEVEL"])
data = dataset.get_curve_data(
    masking_frequency=masking_frequency,
    probe_level=probe_level,
    masking_level=masking_level)
actual_frequencies, actual_amplitudes = zip(*data)

model_class = model.Model(masking_frequency, probe_level, masking_level)

def calculate_model_output(inputs: List[float], pars: List[float]) -> float:
  model_vars = inputs + model_class.parameters_from_learned(pars)
  output = model_class.function(*model_vars)
  return output


def plot_simplex(current_vars: List[float], best_vars: List[float],
                 iteration: int, current_loss: float, best_loss: float,
                 highest_y: float, save_path: str) -> str:
  predicted_amplitudes = []
  best_amplitudes = []
  for frequency in actual_frequencies:
    current_inputs = [frequency
                     ] + model_class.parameters_from_learned(current_vars)
    best_inputs = [frequency] + model_class.parameters_from_learned(best_vars)
    predicted_amplitudes.append(model_class.function(*current_inputs))
    best_amplitudes.append(model_class.function(*best_inputs))
  error_lines = []
  for f, ampl_tuple in zip(actual_frequencies,
                           zip(actual_amplitudes, predicted_amplitudes)):
    pair = [(f, ampl_tuple[0]), (f, ampl_tuple[1])]
    error_lines.append(pair)
  linecoll = matcoll.LineCollection(error_lines, colors="k")
  # Plot fitted data
  _, ax = plt.subplots()
  ax.axvline(model_class.masker_frequency_bark, c="c", label="Fixed Loc")
  ax.scatter(actual_frequencies, actual_amplitudes, c="b", label="Actual")
  ax.scatter(
      actual_frequencies,
      predicted_amplitudes,
      c="r",
      label="Current Predicted")
  ax.scatter(actual_frequencies, best_amplitudes, c="g", label="Best Predicted")
  ax.add_collection(linecoll)
  ax.legend()
  plt.rcParams.update({"font.size": 8})
  title = "Nelder-Mead It. %d\n Current Loss: %.2f, " % (
      iteration, current_loss) + model_class.parameter_repr(
          current_vars) + ", \nBest Loss: %.2f " % (
              best_loss) + model_class.parameter_repr(best_vars)
  plt.ylim(0, highest_y)
  plt.title(title)
  filename = os.path.join(save_path,
                          "fitted_simplex_it{}.png".format(iteration))
  plt.savefig(filename)
  return filename


def plot_loss_curve(losses: List[float], save_path: str) -> str:
  _, ax = plt.subplots()
  ax.plot(list(range(0, len(losses))), losses)
  filename = os.path.join(save_path, "loss_curve.png")
  plt.savefig(filename)
  return filename


def parse_logs(
    path_to_logs: str) -> Tuple[List[float], List[float], float, float]:
  with open(path_to_logs, "r") as infile:
    losses = []
    learned_vars = []
    for line in infile:
      splitted_line = line.split(",")
      loss = float(splitted_line[0])
      losses.append(loss)
      current_vars = [float(var.strip()) for var in splitted_line[1:]]
      learned_vars.append(current_vars)
  return losses, learned_vars


def plot_development(plot_every_n: int, losses: List[float],
                     learned_vars: List[float], logs_dir: str) -> List[str]:
  # Find the highest y for on the y-axis of the plot.
  highest_y = max(actual_amplitudes) + 1

  # Plot progress
  best_loss = float("inf")
  best_vars = [None] * len(learned_vars)
  filenames = []
  for i, (loss, current_vars) in enumerate(zip(losses, learned_vars)):
    if loss < best_loss:
      best_loss = loss
      best_vars = current_vars
    if i % plot_every_n == 0:
      filename = plot_simplex(current_vars, best_vars, i, loss, best_loss,
                              highest_y, logs_dir)
      filenames.append(filename)
  return filenames


def make_gif(filenames: List[str], logs_dir: str):
  # Make a gif of the action sequence.
  images = []
  for filename in filenames:
    images.append(imageio.imread(filename))
  imageio.mimsave(os.path.join(logs_dir, "movie.gif"), images, fps=2)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  log_path = os.path.join(FLAGS.logs_dir, FLAGS.logs_file)
  losses, learned_vars = parse_logs(log_path)
  min_loss_idx = losses.index(min(losses))
  best_learned_vars = learned_vars[min_loss_idx]

  _ = plot_loss_curve(losses, FLAGS.logs_dir)
  filenames = plot_development(FLAGS.plot_every_n_iterations, losses,
                               learned_vars, FLAGS.logs_dir)
  make_gif(filenames, FLAGS.logs_dir)


if __name__ == "__main__":
  app.run(main)
