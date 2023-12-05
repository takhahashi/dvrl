# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def corrupt_label(y_train, noise_rate):
  """Corrupts training labels.

  Args:
    y_train: training labels
    noise_rate: input noise ratio

  Returns:
    corrupted_y_train: corrupted training labels
    noise_idx: corrupted index
  """
  y_set = list(set(y_train))

  # Sets noise_idx
  temp_idx = np.random.permutation(len(y_train))
  noise_idx = temp_idx[:int(len(y_train) * noise_rate)]

  # Corrupts label
  corrupted_y_train = y_train[:] # リスト型のメモリ共有的なものを避けるための記述。copy.deepcopyを使うほうが一般的かも

  for itt in noise_idx:
    temp_y_set = y_set[:]
  
    print("check temp_y_set", temp_y_set)
    print("check y_train[itt]", y_train[itt])

    # 直接ラベルを変更するコードにした
    rand_y = np.random.choice(y_set)
    corrupted_y_train[itt] = rand_y
    print("finish", temp_y_set)

  return corrupted_y_train, noise_idx

# 元コード
  # y_set = list(set(y_train))

  # # Sets noise_idx
  # temp_idx = np.random.permutation(len(y_train))
  # noise_idx = temp_idx[:int(len(y_train) * noise_rate)]

  # # Corrupts label
  # corrupted_y_train = y_train[:]

  # for itt in noise_idx:
  #   temp_y_set = y_set[:]
  #   del temp_y_set[y_train[itt]]
  #   rand_idx = np.random.randint(len(y_set) - 1)
  #   corrupted_y_train[itt] = temp_y_set[rand_idx]

  # return corrupted_y_train, noise_idx
