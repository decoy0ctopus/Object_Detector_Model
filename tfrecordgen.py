import numpy as np
import os

import pandas as pd

from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

import tensorflow as tf
assert tf.__version__.startswith('2')



tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

label_map={1: 'civilian aircraft',
2:
'civilian car',

3:
'military aircraft',
4:
'military helicopter',

5:
'military tank',

6:
'military truck',
}

spec = model_spec.get('efficientdet_lite2')

train_data = object_detector.DataLoader.from_pascal_voc("data/Images/trg", "data/Images/trg", label_map)
test_data = object_detector.DataLoader.from_pascal_voc("data/Images/test", "data/Images/test", label_map)
valid_data = object_detector.DataLoader.from_pascal_voc("data/Images/valid", "data/Images/valid", label_map)

#train_data, validation_data, test_data = object_detector.DataLoader.from_csv("label.csv")

model = object_detector.create(train_data, model_spec=spec, batch_size=20, train_whole_model=True, validation_data=valid_data)

model.evaluate(test_data)

model.export(export_dir='.')

model.evaluate_tflite('model.tflite', test_data)