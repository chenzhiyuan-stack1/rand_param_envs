# coding=utf-8
# Copyright 2020 The Google Research Authors.
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
"""A gin-config class for locomotion_gym_env.

This should be identical to locomotion_gym_config.proto.
"""
import attr
import typing

@attr.s
class ScalarField(object):
  """A named scalar space with bounds."""
  name = attr.ib(type=str)
  upper_bound = attr.ib(type=float)
  lower_bound = attr.ib(type=float)
