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

"""This file implements an accurate motor model."""

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
import numpy as np

from rand_param_envs.utilities import robot_config

VOLTAGE_CLIPPING = 50
OBSERVED_TORQUE_LIMIT = 5.7
MOTOR_VOLTAGE = 16.0
MOTOR_RESISTANCE = 0.186
MOTOR_TORQUE_CONSTANT = 0.0954
MOTOR_VISCOUS_DAMPING = 0
MOTOR_SPEED_LIMIT = MOTOR_VOLTAGE / (MOTOR_VISCOUS_DAMPING +
                                     MOTOR_TORQUE_CONSTANT)
NUM_MOTORS = 8
MOTOR_POS_LB = 0.5
MOTOR_POS_UB = 2.5


class MotorModel(object):
  """The accurate motor model, which is based on the physics of DC motors.

  The motor model support two types of control: position control and torque
  control. In position control mode, a desired motor angle is specified, and a
  torque is computed based on the internal motor model. When the torque control
  is specified, a pwm signal in the range of [-1.0, 1.0] is converted to the
  torque.

  The internal motor model takes the following factors into consideration:
  pd gains, viscous friction, back-EMF voltage and current-torque profile.
  """
  def __init__(self,
               kp=1.2,
               kd=0,
               torque_limits=None,
               motor_control_mode=robot_config.MotorControlMode.POSITION):
    self._kp = kp
    self._kd = kd
    self._torque_limits = torque_limits
    self._motor_control_mode = motor_control_mode
    self._resistance = MOTOR_RESISTANCE
    self._voltage = MOTOR_VOLTAGE
    self._torque_constant = MOTOR_TORQUE_CONSTANT
    self._viscous_damping = MOTOR_VISCOUS_DAMPING
    self._current_table = [0, 10, 20, 30, 40, 50, 60]
    self._torque_table = [0, 1, 1.9, 2.45, 3.0, 3.25, 3.5]
    self._strength_ratios = [1.0] * NUM_MOTORS

  def set_strength_ratios(self, ratios):
    """Set the strength of each motors relative to the default value.

    Args:
      ratios: The relative strength of motor output. A numpy array ranging from
        0.0 to 1.0.
    """
    self._strength_ratios = np.array(ratios)

  def set_motor_gains(self, kp, kd):
    """Set the gains of all motors.

    These gains are PD gains for motor positional control. kp is the
    proportional gain and kd is the derivative gain.

    Args:
      kp: proportional gain of the motors.
      kd: derivative gain of the motors.
    """
    self._kp = kp
    self._kd = kd

  def set_voltage(self, voltage):
    self._voltage = voltage

  def get_voltage(self):
    return self._voltage

  def set_viscous_damping(self, viscous_damping):
    self._viscous_damping = viscous_damping

  def get_viscous_dampling(self):
    return self._viscous_damping

  def convert_to_torque(self,
                        motor_commands,
                        motor_angle,
                        motor_velocity,
                        true_motor_velocity,
                        motor_control_mode=None):
    """Convert the commands (position control or pwm control) to torque.

    Args:
      motor_commands: The desired motor angle if the motor is in position
        control mode. The pwm signal if the motor is in torque control mode.
      motor_angle: The motor angle observed at the current time step. It is
        actually the true motor angle observed a few milliseconds ago (pd
        latency).
      motor_velocity: The motor velocity observed at the current time step, it
        is actually the true motor velocity a few milliseconds ago (pd latency).
      true_motor_velocity: The true motor velocity. The true velocity is used to
        compute back EMF voltage and viscous damping.
      motor_control_mode: A MotorControlMode enum.

    Returns:
      actual_torque: The torque that needs to be applied to the motor.
      observed_torque: The torque observed by the sensor.
    """
    if not motor_control_mode:
      motor_control_mode = self._motor_control_mode

    if (motor_control_mode is robot_config.MotorControlMode.TORQUE) or (
        motor_control_mode is robot_config.MotorControlMode.HYBRID):
      raise ValueError("{} is not a supported motor control mode".format(
          motor_control_mode))

    kp = self._kp
    kd = self._kd

    if motor_control_mode is robot_config.MotorControlMode.PWM:
      # The following implements a safety controller that softly enforces the
      # joint angles to remain within safe region: If PD controller targeting
      # the positive (negative) joint limit outputs a negative (positive)
      # signal, the corresponding joint violates the joint constraint, so
      # we should add the PD output to motor_command to bring it back to the
      # safe region.
      pd_max = -1 * kp * (motor_angle -
                          MOTOR_POS_UB) - kd / 2. * motor_velocity
      pd_min = -1 * kp * (motor_angle -
                          MOTOR_POS_LB) - kd / 2. * motor_velocity
      pwm = motor_commands + np.minimum(pd_max, 0) + np.maximum(pd_min, 0)
    else:
      pwm = -1 * kp * (motor_angle - motor_commands) - kd * motor_velocity
    pwm = np.clip(pwm, -1.0, 1.0)
    return self._convert_to_torque_from_pwm(pwm, true_motor_velocity)

  def _convert_to_torque_from_pwm(self, pwm, true_motor_velocity):
    """Convert the pwm signal to torque.

    Args:
      pwm: The pulse width modulation.
      true_motor_velocity: The true motor velocity at the current moment. It is
        used to compute the back EMF voltage and the viscous damping.

    Returns:
      actual_torque: The torque that needs to be applied to the motor.
      observed_torque: The torque observed by the sensor.
    """
    observed_torque = np.clip(
        self._torque_constant *
        (np.asarray(pwm) * self._voltage / self._resistance),
        -OBSERVED_TORQUE_LIMIT, OBSERVED_TORQUE_LIMIT)
    if self._torque_limits is not None:
      observed_torque = np.clip(observed_torque, -1.0 * self._torque_limits,
                                self._torque_limits)

    # Net voltage is clipped at 50V by diodes on the motor controller.
    voltage_net = np.clip(
        np.asarray(pwm) * self._voltage -
        (self._torque_constant + self._viscous_damping) *
        np.asarray(true_motor_velocity), -VOLTAGE_CLIPPING, VOLTAGE_CLIPPING)
    current = voltage_net / self._resistance
    current_sign = np.sign(current)
    current_magnitude = np.absolute(current)
    # Saturate torque based on empirical current relation.
    actual_torque = np.interp(current_magnitude, self._current_table,
                              self._torque_table)
    actual_torque = np.multiply(current_sign, actual_torque)
    actual_torque = np.multiply(self._strength_ratios, actual_torque)
    if self._torque_limits is not None:
      actual_torque = np.clip(actual_torque, -1.0 * self._torque_limits,
                              self._torque_limits)
    return actual_torque, observed_torque
