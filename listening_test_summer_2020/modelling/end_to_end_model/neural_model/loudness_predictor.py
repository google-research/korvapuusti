# Lint as: python3
"""Pipeline for training and evaluating a loudness predictor."""
import os
from typing import List

from absl import app
from absl import flags
from absl import logging
import tensorflow.compat.v2 as tf

from data_processing import get_datasets, get_testdata
from evaluate import evaluate, write_predictions
from model import LoudnessPredictor
import train
import helpers

tf.enable_v2_behavior()

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "mode",
    "train",
    "Whether to train, test, or predict.")
flags.DEFINE_string(
    "loudness_traindata_proto_file_pattern",
    "../../data/preprocessed_data/all_dl_data/traindataraw_*",
    "Proto file to read loudness data from.")
flags.DEFINE_string(
    "extra_loudness_traindata_proto_file_pattern",
    "../../data/preprocessed_data/all_dl_data/extratraindataraw_*",
    "Proto file to read loudness data from.")
flags.DEFINE_string(
    "loudness_testdata_proto_file_pattern",
    "../../data/preprocessed_data/all_dl_data/traindataraw_*",
    "Proto file to read loudness data from.")
flags.DEFINE_bool("use_carfac", True,
                  "Whether to use CARFAC features or not.")
flags.DEFINE_string(
    "logs_dir",
    "logs/loudness_predictor",
    "Directory to put summary logs.")
flags.DEFINE_string(
    "load_from_checkpoint",
    "",
    "Which checkpoint from log dir to load when starting (e.g., cp.ckpt).")
flags.DEFINE_integer(
    "num_rows_channel_kernel",
    5,
    "Number of rows on the kernel that will convolve the input across "
    "CARFAC channels; it will summarize num_rows_channel_kernel channels.")
flags.DEFINE_integer(
    "num_cols_channel_kernel",
    1,
    "Number of cols on the kernel that will convolve the input across "
    "CARFAC channels; it will summarize num_cols_channel_kernel freq. bins.")
flags.DEFINE_integer(
    "num_filters_channels",
    5,
    "Number of filters when convolving channels across bins.")
flags.DEFINE_integer(
    "num_rows_bin_kernel",
    1,
    "Number of rows on the kernel that will convolve the input across "
    "frequency bins; it will summarize num_rows_bin_kernel CARFAC channels.")
flags.DEFINE_integer(
    "num_cols_bin_kernel",
    1,
    "Number of rows on the kernel that will convolve the input across "
    "frequency bins; it will summarize num_cols_bin_kernel frequency bins.")
flags.DEFINE_integer(
    "num_filters_bins",
    5,
    "Number of filters when convolving bins across channels.")
flags.DEFINE_integer(
    "batch_size",
    1,
    "Batch size when training.")
flags.DEFINE_integer(
    "epochs",
    150,
    "Number of epochs (full passes of the training data) to train the model.")
flags.DEFINE_string(
    "save_model_to_dir",
    "saved_model",
    "Destination directory for saving the model before early exit.")
flags.DEFINE_string(
    "load_model_from_file",
    "",
    "A saved model to eval once and then return.")
flags.DEFINE_string(
    "save_predictions_file",
    "predictions.txt",
    "A saved model to eval once and then return.")
flags.DEFINE_integer(
    "seed",
    4,
    "TensorFlow random seed.")
flags.DEFINE_float("learning_rate", 1e-3, "The initial learning rate for Adam.")
flags.DEFINE_float("dropout_p", 0., "Dropout to apply to the fc layers.")



def main(argv):

  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  tf.random.set_seed(FLAGS.seed)

  if FLAGS.loudness_traindata_proto_file_pattern is None:
    raise app.UsageError("Must provide --loudness_data_proto_file_pattern.")

  log_dir = os.path.join(
      os.path.dirname(os.path.realpath(__file__)), FLAGS.logs_dir)
  if not os.path.exists(log_dir):
    os.mkdir(log_dir)

  logging.info("TensorFlow seed: %d", FLAGS.seed)

  input_shape = None
  if FLAGS.mode == "test":
    raise NotImplementedError("Did not implement mode test.")
    data = get_datasets(FLAGS.loudness_testdata_proto_file_pattern, 1,
                        carfac=FLAGS.use_carfac)
    logging.info("Created testing datasets")
    model = tf.keras.models.load_model(FLAGS.load_model_from_file)
    logging.info("Loaded model")
  elif FLAGS.mode == "train":
    data = get_datasets(FLAGS.loudness_traindata_proto_file_pattern,
                        FLAGS.batch_size, carfac=FLAGS.use_carfac,
                        extra_file_pattern=FLAGS.extra_loudness_traindata_proto_file_pattern)
    frequency_bins = None
    for example in data["train"].take(1):
      input_example, target_example = example
      input_shape = input_example.shape
      carfac_channels = input_example.shape[1]
      frequency_bins = input_example.shape[2]
    logging.info("Created model")
  elif FLAGS.mode == "eval_once":
    data = get_testdata(FLAGS.loudness_testdata_proto_file_pattern,
                        carfac=FLAGS.use_carfac)
    frequency_bins = None
    for example in data["test"].take(1):
      input_example, target_example, _ = example
      input_shape = input_example.shape
      carfac_channels = input_example.shape[1]
      frequency_bins = input_example.shape[2]
  model = LoudnessPredictor(
        frequency_bins=frequency_bins,
        carfac_channels=carfac_channels,
        num_rows_channel_kernel=FLAGS.num_rows_channel_kernel,
        num_cols_channel_kernel=FLAGS.num_cols_channel_kernel,
        num_filters_channels=FLAGS.num_filters_channels,
        num_rows_bin_kernel=FLAGS.num_rows_bin_kernel,
        num_cols_bin_kernel=FLAGS.num_cols_bin_kernel,
        num_filters_bins=FLAGS.num_filters_bins,
        dropout_p=FLAGS.dropout_p,
        use_channels=FLAGS.use_carfac,
        seed=FLAGS.seed)
  if FLAGS.load_from_checkpoint:
    path_to_load = os.path.join(log_dir, FLAGS.load_from_checkpoint)
    logging.info("Attempting to load model from %s", path_to_load)
    loaded = False
    try:
      model.load_weights(path_to_load)
      loaded = True
      logging.info("Loaded model")
    except Exception as err:
      logging.info("Unable to load log dir checkpoint %s, trying "
                   "'load_from_checkpoint' flag: %s", path_to_load, err)
      path_to_load = FLAGS.load_from_checkpoint
      try:
        model.load_weights(path_to_load)
        loaded = True
      except Exception as err:
        logging.info("Unable to load flag checkpoint %s: %s",
                     path_to_load, err)
  else:
    loaded = False

  example_image_batch = []
  if FLAGS.mode == "train":
    data_key = "train"
    for example in data[data_key].take(4):
      input_example, target = example
      input_shape = input_example.shape
      tf.print("(start train) input shape: ", input_shape)
      tf.print("(start train) target phons shape: ", target.shape)
      input_example = tf.expand_dims(input_example[0], axis=0)
      example_image_batch.append(
          [input_example, target])

  elif FLAGS.mode == "eval_once":
    data_key = "test"
    for example in data[data_key].take(4):
      input_example, target, _ = example
      input_shape = input_example.shape
      tf.print("(start eval) input shape: ", input_shape)
      tf.print("(start eval) target phons shape: ", target.shape)
      input_example = tf.expand_dims(input_example[0], axis=0)
      example_image_batch.append(
          [input_example, target])

  callbacks = [helpers.StepIncrementingCallback()]
  callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                  histogram_freq=1,
                                                  update_freq="batch",
                                                  write_graph=True))
  model.build(input_shape)
  logging.info("Model summary")
  model.summary()

  if FLAGS.extra_loudness_traindata_proto_file_pattern:
    extra_data = True
  else:
    extra_data = False
  save_ckpt = log_dir + "/cp_carfac{}_extradata{}".format(
      FLAGS.use_carfac, extra_data) + "_{epoch:04d}.ckpt"
  logging.info("Save checkpoint to: %s" % save_ckpt)
  callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=save_ckpt,
                                                      save_weights_only=True,
                                                      verbose=1,
                                                      period=5))

  if FLAGS.mode == "train":
    logging.info("Starting training for %d epochs" % FLAGS.epochs)
    if FLAGS.extra_loudness_traindata_proto_file_pattern:
      steps_per_epoch = (317 + 639) // FLAGS.batch_size
    else:
      steps_per_epoch = 317 // FLAGS.batch_size
    train(model, data["train"], data["validate"], FLAGS.learning_rate,
          FLAGS.epochs, steps_per_epoch, callbacks)
  elif FLAGS.mode == "test":
    raise NotImplementedError("Mode test not implemented.")
    evaluate(model, data["test"], batch_size=FLAGS.eval_batch_size)
  elif FLAGS.mode == "eval_once":
    if not loaded:
      raise ValueError("Trying to eval. a model with unitialized weights.")
    save_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), log_dir)
    write_predictions(model, data["test"], batch_size=1,
                      save_directory=save_dir,
                      save_file=FLAGS.save_predictions_file)
    return
  else:
    raise ValueError("Specified value for '--mode' (%s) unknown", FLAGS.mode)

if __name__ == "__main__":
  app.run(main)
