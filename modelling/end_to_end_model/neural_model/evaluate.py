# Lint as: python3
"""Predict all examples in a provided dataset and write them to a file."""
import time
from absl import logging
import tensorflow.compat.v2 as tf
import os

import LoudnessPredictor
from train import multi_output_loss, MSEMetric, AccuracyMetric

tf.enable_v2_behavior()


def evaluate(model: LoudnessPredictor, data: tf.data.Dataset,
             batch_size: int):
  start = time.time()
  optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
  model.compile(optimizer=optimizer,
                loss=multi_output_loss,
                metrics=[MSEMetric()])
  logging.info('Compiled model')
  results = model.evaluate(data, batch_size=batch_size, return_dict=True)
  end = time.time()
  logging.info("Predicted examples in %f s", end - start)
  for metric, result in results.items():
    logging.info("%s: %.5f", metric, result)
  return


def write_predictions(model: LoudnessPredictor, data: tf.data.Dataset,
                      batch_size: int, save_directory: str, save_file: str):
  optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
  model.compile(optimizer=optimizer,
                loss=multi_output_loss,
                metrics=[MSEMetric(), AccuracyMetric()])
  logging.info('Compiled model')
  predictions = []
  total_error = 0
  total_baseline_error = 0
  num_predictions = 0
  with open(os.path.join(save_directory, save_file), "w") as infile:
    infile.write("frequencies;spls;probefrequency;probelevel;probeloudness;targetmasking;predictedmasking\n")
    for i, example in data.enumerate(start=0):
      input_example, target_phons, rest = example
      predicted_phons = model.predict(input_example)
      error = ((target_phons / 10) - (predicted_phons / 10))**2
      baseline_error = ((target_phons / 10) - (0 / 10))**2
      total_error += error[0][0]
      total_baseline_error += baseline_error[0][0]
      num_predictions += 1
      logging.info('Prediction %d, actual masking %.4f, predicted masking %.4f',
                   i, target_phons.numpy()[0][0], predicted_phons[0][0])
      infile.write(",".join([str(f) for f in rest["frequencies"].numpy()[0]]))
      infile.write(";")
      infile.write(",".join([str(l) for l in rest["spls"].numpy()[0]]))
      infile.write(";")
      infile.write(str(rest["probefrequency"].numpy()[0][0]))
      infile.write(";")
      infile.write(str(rest["probelevel"].numpy()[0][0]))
      infile.write(";")
      infile.write(str(rest["probeloudness"].numpy()[0][0]))
      infile.write(";")
      infile.write(str(target_phons.numpy()[0][0]))
      infile.write(";")
      infile.write(str(predicted_phons[0][0]))
      infile.write("\n")
    infile.write("\n")
    infile.write("Baseline MSE: " + str(total_baseline_error / num_predictions))
    infile.write("\n")
    infile.write("MSE: " + str(total_error / num_predictions))
  return
