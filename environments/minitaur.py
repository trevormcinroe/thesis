"""
Original: https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/bullet/minitaur_gym_env.py
	Also flattening out the imports

TODO: Add in the ability to take an image from the environment that follows the minitaur along its path


"""

import inspect
import os

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import math
import time
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pybullet
from pybullet_utils import bullet_client as bc
# from . import minitaur
import os
import pybullet_data
# from . import minitaur_env_randomizer
from pkg_resources import parse_version
import copy
import abc
import random

"""MOTOR CLASS USED TO MOVE THE MINITAUR"""
VOLTAGE_CLIPPING = 50
OBSERVED_TORQUE_LIMIT = 5.7
MOTOR_VOLTAGE = 16.0
MOTOR_RESISTANCE = 0.186
MOTOR_TORQUE_CONSTANT = 0.0954
MOTOR_VISCOUS_DAMPING = 0
MOTOR_SPEED_LIMIT = MOTOR_VOLTAGE / (MOTOR_VISCOUS_DAMPING + MOTOR_TORQUE_CONSTANT)


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

	def __init__(self, torque_control_enabled=False, kp=1.2, kd=0):
		self._torque_control_enabled = torque_control_enabled
		self._kp = kp
		self._kd = kd
		self._resistance = MOTOR_RESISTANCE
		self._voltage = MOTOR_VOLTAGE
		self._torque_constant = MOTOR_TORQUE_CONSTANT
		self._viscous_damping = MOTOR_VISCOUS_DAMPING
		self._current_table = [0, 10, 20, 30, 40, 50, 60]
		self._torque_table = [0, 1, 1.9, 2.45, 3.0, 3.25, 3.5]

	def set_voltage(self, voltage):
		self._voltage = voltage

	def get_voltage(self):
		return self._voltage

	def set_viscous_damping(self, viscous_damping):
		self._viscous_damping = viscous_damping

	def get_viscous_dampling(self):
		return self._viscous_damping

	def convert_to_torque(self, motor_commands, current_motor_angle, current_motor_velocity):
		"""Convert the commands (position control or torque control) to torque.
		Args:
		  motor_commands: The desired motor angle if the motor is in position
			control mode. The pwm signal if the motor is in torque control mode.
		  current_motor_angle: The motor angle at the current time step.
		  current_motor_velocity: The motor velocity at the current time step.
		Returns:
		  actual_torque: The torque that needs to be applied to the motor.
		  observed_torque: The torque observed by the sensor.
		"""
		if self._torque_control_enabled:
			pwm = motor_commands
		else:
			pwm = (-self._kp * (current_motor_angle - motor_commands) -
				   self._kd * current_motor_velocity)
		pwm = np.clip(pwm, -1.0, 1.0)
		return self._convert_to_torque_from_pwm(pwm, current_motor_velocity)

	def _convert_to_torque_from_pwm(self, pwm, current_motor_velocity):
		"""Convert the pwm signal to torque.
		Args:
		  pwm: The pulse width modulation.
		  current_motor_velocity: The motor velocity at the current time step.
		Returns:
		  actual_torque: The torque that needs to be applied to the motor.
		  observed_torque: The torque observed by the sensor.
		"""
		observed_torque = np.clip(self._torque_constant * (pwm * self._voltage / self._resistance),
								  -OBSERVED_TORQUE_LIMIT, OBSERVED_TORQUE_LIMIT)

		# Net voltage is clipped at 50V by diodes on the motor controller.
		voltage_net = np.clip(
			pwm * self._voltage -
			(self._torque_constant + self._viscous_damping) * current_motor_velocity,
			-VOLTAGE_CLIPPING, VOLTAGE_CLIPPING)
		current = voltage_net / self._resistance
		current_sign = np.sign(current)
		current_magnitude = np.absolute(current)

		# Saturate torque based on empirical current relation.
		actual_torque = np.interp(current_magnitude, self._current_table, self._torque_table)
		actual_torque = np.multiply(current_sign, actual_torque)
		return actual_torque, observed_torque


"""MINITAUR CLASS FROM GHOST ROBOTICS"""
INIT_POSITION = [0, 0, .2]
INIT_ORIENTATION = [0, 0, 0, 1]
KNEE_CONSTRAINT_POINT_RIGHT = [0, 0.005, 0.2]
KNEE_CONSTRAINT_POINT_LEFT = [0, 0.01, 0.2]
OVERHEAT_SHUTDOWN_TORQUE = 2.45
OVERHEAT_SHUTDOWN_TIME = 1.0
LEG_POSITION = ["front_left", "back_left", "front_right", "back_right"]
MOTOR_NAMES = [
	"motor_front_leftL_joint", "motor_front_leftR_joint", "motor_back_leftL_joint",
	"motor_back_leftR_joint", "motor_front_rightL_joint", "motor_front_rightR_joint",
	"motor_back_rightL_joint", "motor_back_rightR_joint"
]
LEG_LINK_ID = [2, 3, 5, 6, 8, 9, 11, 12, 15, 16, 18, 19, 21, 22, 24, 25]
MOTOR_LINK_ID = [1, 4, 7, 10, 14, 17, 20, 23]
FOOT_LINK_ID = [3, 6, 9, 12, 16, 19, 22, 25]
BASE_LINK_ID = -1


class Minitaur(object):
	"""The minitaur class that simulates a quadruped robot from Ghost Robotics.
	"""

	def __init__(self,
				 pybullet_client,
				 urdf_root=os.path.join(os.path.dirname(__file__), "../data"),
				 time_step=0.01,
				 self_collision_enabled=False,
				 motor_velocity_limit=np.inf,
				 pd_control_enabled=False,
				 accurate_motor_model_enabled=False,
				 motor_kp=1.0,
				 motor_kd=0.02,
				 torque_control_enabled=False,
				 motor_overheat_protection=False,
				 on_rack=False,
				 kd_for_pd_controllers=0.3):
		"""Constructs a minitaur and reset it to the initial states.
		Args:
		  pybullet_client: The instance of BulletClient to manage different
			simulations.
		  urdf_root: The path to the urdf folder.
		  time_step: The time step of the simulation.
		  self_collision_enabled: Whether to enable self collision.
		  motor_velocity_limit: The upper limit of the motor velocity.
		  pd_control_enabled: Whether to use PD control for the motors.
		  accurate_motor_model_enabled: Whether to use the accurate DC motor model.
		  motor_kp: proportional gain for the accurate motor model
		  motor_kd: derivative gain for the acurate motor model
		  torque_control_enabled: Whether to use the torque control, if set to
			False, pose control will be used.
		  motor_overheat_protection: Whether to shutdown the motor that has exerted
			large torque (OVERHEAT_SHUTDOWN_TORQUE) for an extended amount of time
			(OVERHEAT_SHUTDOWN_TIME). See ApplyAction() in minitaur.py for more
			details.
		  on_rack: Whether to place the minitaur on rack. This is only used to debug
			the walking gait. In this mode, the minitaur's base is hanged midair so
			that its walking gait is clearer to visualize.
		  kd_for_pd_controllers: kd value for the pd controllers of the motors.
		"""
		self.num_motors = 8
		self.num_legs = int(self.num_motors / 2)
		self._pybullet_client = pybullet_client
		self._urdf_root = urdf_root
		self._self_collision_enabled = self_collision_enabled
		self._motor_velocity_limit = motor_velocity_limit
		self._pd_control_enabled = pd_control_enabled
		self._motor_direction = [-1, -1, -1, -1, 1, 1, 1, 1]
		self._observed_motor_torques = np.zeros(self.num_motors)
		self._applied_motor_torques = np.zeros(self.num_motors)
		self._max_force = 3.5
		self._accurate_motor_model_enabled = accurate_motor_model_enabled
		self._torque_control_enabled = torque_control_enabled
		self._motor_overheat_protection = motor_overheat_protection
		self._on_rack = on_rack
		if self._accurate_motor_model_enabled:
			self._kp = motor_kp
			self._kd = motor_kd
			self._motor_model = MotorModel(torque_control_enabled=self._torque_control_enabled,
										   kp=self._kp,
										   kd=self._kd)
		elif self._pd_control_enabled:
			self._kp = 8
			self._kd = kd_for_pd_controllers
		else:
			self._kp = 1
			self._kd = 1
		self.time_step = time_step
		self.Reset()

	def _RecordMassInfoFromURDF(self):
		self._base_mass_urdf = self._pybullet_client.getDynamicsInfo(self.quadruped, BASE_LINK_ID)[0]
		self._leg_masses_urdf = []
		self._leg_masses_urdf.append(
			self._pybullet_client.getDynamicsInfo(self.quadruped, LEG_LINK_ID[0])[0])
		self._leg_masses_urdf.append(
			self._pybullet_client.getDynamicsInfo(self.quadruped, MOTOR_LINK_ID[0])[0])

	def _BuildJointNameToIdDict(self):
		num_joints = self._pybullet_client.getNumJoints(self.quadruped)
		self._joint_name_to_id = {}
		for i in range(num_joints):
			joint_info = self._pybullet_client.getJointInfo(self.quadruped, i)
			self._joint_name_to_id[joint_info[1].decode("UTF-8")] = joint_info[0]

	def _BuildMotorIdList(self):
		self._motor_id_list = [self._joint_name_to_id[motor_name] for motor_name in MOTOR_NAMES]

	def Reset(self, reload_urdf=True):
		"""Reset the minitaur to its initial states.
		Args:
		  reload_urdf: Whether to reload the urdf file. If not, Reset() just place
			the minitaur back to its starting position.
		"""
		if reload_urdf:
			if self._self_collision_enabled:
				self.quadruped = self._pybullet_client.loadURDF(
					"%s/quadruped/minitaur.urdf" % self._urdf_root,
					INIT_POSITION,
					flags=self._pybullet_client.URDF_USE_SELF_COLLISION)
			else:
				self.quadruped = self._pybullet_client.loadURDF(
					"%s/quadruped/minitaur.urdf" % self._urdf_root, INIT_POSITION)
			self._BuildJointNameToIdDict()
			self._BuildMotorIdList()
			self._RecordMassInfoFromURDF()
			self.ResetPose(add_constraint=True)
			if self._on_rack:
				self._pybullet_client.createConstraint(self.quadruped, -1, -1, -1,
													   self._pybullet_client.JOINT_FIXED, [0, 0, 0],
													   [0, 0, 0], [0, 0, 1])
		else:
			self._pybullet_client.resetBasePositionAndOrientation(self.quadruped, INIT_POSITION,
																  INIT_ORIENTATION)
			self._pybullet_client.resetBaseVelocity(self.quadruped, [0, 0, 0], [0, 0, 0])
			self.ResetPose(add_constraint=False)

		self._overheat_counter = np.zeros(self.num_motors)
		self._motor_enabled_list = [True] * self.num_motors

	def _SetMotorTorqueById(self, motor_id, torque):
		self._pybullet_client.setJointMotorControl2(bodyIndex=self.quadruped,
													jointIndex=motor_id,
													controlMode=self._pybullet_client.TORQUE_CONTROL,
													force=torque)

	def _SetDesiredMotorAngleById(self, motor_id, desired_angle):
		self._pybullet_client.setJointMotorControl2(bodyIndex=self.quadruped,
													jointIndex=motor_id,
													controlMode=self._pybullet_client.POSITION_CONTROL,
													targetPosition=desired_angle,
													positionGain=self._kp,
													velocityGain=self._kd,
													force=self._max_force)

	def _SetDesiredMotorAngleByName(self, motor_name, desired_angle):
		self._SetDesiredMotorAngleById(self._joint_name_to_id[motor_name], desired_angle)

	def ResetPose(self, add_constraint):
		"""Reset the pose of the minitaur.
		Args:
		  add_constraint: Whether to add a constraint at the joints of two feet.
		"""
		for i in range(self.num_legs):
			self._ResetPoseForLeg(i, add_constraint)

	def _ResetPoseForLeg(self, leg_id, add_constraint):
		"""Reset the initial pose for the leg.
		Args:
		  leg_id: It should be 0, 1, 2, or 3, which represents the leg at
			front_left, back_left, front_right and back_right.
		  add_constraint: Whether to add a constraint at the joints of two feet.
		"""
		knee_friction_force = 0
		half_pi = math.pi / 2.0
		knee_angle = -2.1834

		leg_position = LEG_POSITION[leg_id]
		self._pybullet_client.resetJointState(self.quadruped,
											  self._joint_name_to_id["motor_" + leg_position +
																	 "L_joint"],
											  self._motor_direction[2 * leg_id] * half_pi,
											  targetVelocity=0)
		self._pybullet_client.resetJointState(self.quadruped,
											  self._joint_name_to_id["knee_" + leg_position +
																	 "L_link"],
											  self._motor_direction[2 * leg_id] * knee_angle,
											  targetVelocity=0)
		self._pybullet_client.resetJointState(self.quadruped,
											  self._joint_name_to_id["motor_" + leg_position +
																	 "R_joint"],
											  self._motor_direction[2 * leg_id + 1] * half_pi,
											  targetVelocity=0)
		self._pybullet_client.resetJointState(self.quadruped,
											  self._joint_name_to_id["knee_" + leg_position +
																	 "R_link"],
											  self._motor_direction[2 * leg_id + 1] * knee_angle,
											  targetVelocity=0)
		if add_constraint:
			self._pybullet_client.createConstraint(
				self.quadruped, self._joint_name_to_id["knee_" + leg_position + "R_link"],
				self.quadruped, self._joint_name_to_id["knee_" + leg_position + "L_link"],
				self._pybullet_client.JOINT_POINT2POINT, [0, 0, 0], KNEE_CONSTRAINT_POINT_RIGHT,
				KNEE_CONSTRAINT_POINT_LEFT)

		if self._accurate_motor_model_enabled or self._pd_control_enabled:
			# Disable the default motor in pybullet.
			self._pybullet_client.setJointMotorControl2(
				bodyIndex=self.quadruped,
				jointIndex=(self._joint_name_to_id["motor_" + leg_position + "L_joint"]),
				controlMode=self._pybullet_client.VELOCITY_CONTROL,
				targetVelocity=0,
				force=knee_friction_force)
			self._pybullet_client.setJointMotorControl2(
				bodyIndex=self.quadruped,
				jointIndex=(self._joint_name_to_id["motor_" + leg_position + "R_joint"]),
				controlMode=self._pybullet_client.VELOCITY_CONTROL,
				targetVelocity=0,
				force=knee_friction_force)

		else:
			self._SetDesiredMotorAngleByName("motor_" + leg_position + "L_joint",
											 self._motor_direction[2 * leg_id] * half_pi)
			self._SetDesiredMotorAngleByName("motor_" + leg_position + "R_joint",
											 self._motor_direction[2 * leg_id + 1] * half_pi)

		self._pybullet_client.setJointMotorControl2(
			bodyIndex=self.quadruped,
			jointIndex=(self._joint_name_to_id["knee_" + leg_position + "L_link"]),
			controlMode=self._pybullet_client.VELOCITY_CONTROL,
			targetVelocity=0,
			force=knee_friction_force)
		self._pybullet_client.setJointMotorControl2(
			bodyIndex=self.quadruped,
			jointIndex=(self._joint_name_to_id["knee_" + leg_position + "R_link"]),
			controlMode=self._pybullet_client.VELOCITY_CONTROL,
			targetVelocity=0,
			force=knee_friction_force)

	def GetBasePosition(self):
		"""Get the position of minitaur's base.
		Returns:
		  The position of minitaur's base.
		"""
		position, _ = (self._pybullet_client.getBasePositionAndOrientation(self.quadruped))
		return position

	def GetBaseOrientation(self):
		"""Get the orientation of minitaur's base, represented as quaternion.
		Returns:
		  The orientation of minitaur's base.
		"""
		_, orientation = (self._pybullet_client.getBasePositionAndOrientation(self.quadruped))
		return orientation

	def GetActionDimension(self):
		"""Get the length of the action list.
		Returns:
		  The length of the action list.
		"""
		return self.num_motors

	def GetObservationUpperBound(self):
		"""Get the upper bound of the observation.
		Returns:
		  The upper bound of an observation. See GetObservation() for the details
			of each element of an observation.
		"""
		upper_bound = np.array([0.0] * self.GetObservationDimension())
		upper_bound[0:self.num_motors] = math.pi  # Joint angle.
		upper_bound[self.num_motors:2 * self.num_motors] = (MOTOR_SPEED_LIMIT)  # Joint velocity.
		upper_bound[2 * self.num_motors:3 * self.num_motors] = (OBSERVED_TORQUE_LIMIT
																)  # Joint torque.
		upper_bound[3 * self.num_motors:] = 1.0  # Quaternion of base orientation.
		return upper_bound

	def GetObservationLowerBound(self):
		"""Get the lower bound of the observation."""
		return -self.GetObservationUpperBound()

	def GetObservationDimension(self):
		"""Get the length of the observation list.
		Returns:
		  The length of the observation list.
		"""
		return len(self.GetObservation())

	def GetObservation(self):
		"""Get the observations of minitaur.
		It includes the angles, velocities, torques and the orientation of the base.
		Returns:
		  The observation list. observation[0:8] are motor angles. observation[8:16]
		  are motor velocities, observation[16:24] are motor torques.
		  observation[24:28] is the orientation of the base, in quaternion form.
		"""
		observation = []
		observation.extend(self.GetMotorAngles().tolist())
		observation.extend(self.GetMotorVelocities().tolist())
		observation.extend(self.GetMotorTorques().tolist())
		observation.extend(list(self.GetBaseOrientation()))
		return observation

	def ApplyAction(self, motor_commands):
		"""Set the desired motor angles to the motors of the minitaur.
		The desired motor angles are clipped based on the maximum allowed velocity.
		If the pd_control_enabled is True, a torque is calculated according to
		the difference between current and desired joint angle, as well as the joint
		velocity. This torque is exerted to the motor. For more information about
		PD control, please refer to: https://en.wikipedia.org/wiki/PID_controller.
		Args:
		  motor_commands: The eight desired motor angles.
		"""
		if self._motor_velocity_limit < np.inf:
			current_motor_angle = self.GetMotorAngles()
			motor_commands_max = (current_motor_angle + self.time_step * self._motor_velocity_limit)
			motor_commands_min = (current_motor_angle - self.time_step * self._motor_velocity_limit)
			motor_commands = np.clip(motor_commands, motor_commands_min, motor_commands_max)

		if self._accurate_motor_model_enabled or self._pd_control_enabled:
			q = self.GetMotorAngles()
			qdot = self.GetMotorVelocities()
			if self._accurate_motor_model_enabled:
				actual_torque, observed_torque = self._motor_model.convert_to_torque(
					motor_commands, q, qdot)
				if self._motor_overheat_protection:
					for i in range(self.num_motors):
						if abs(actual_torque[i]) > OVERHEAT_SHUTDOWN_TORQUE:
							self._overheat_counter[i] += 1
						else:
							self._overheat_counter[i] = 0
						if (self._overheat_counter[i] > OVERHEAT_SHUTDOWN_TIME / self.time_step):
							self._motor_enabled_list[i] = False

				# The torque is already in the observation space because we use
				# GetMotorAngles and GetMotorVelocities.
				self._observed_motor_torques = observed_torque

				# Transform into the motor space when applying the torque.
				self._applied_motor_torque = np.multiply(actual_torque, self._motor_direction)

				for motor_id, motor_torque, motor_enabled in zip(self._motor_id_list,
																 self._applied_motor_torque,
																 self._motor_enabled_list):
					if motor_enabled:
						self._SetMotorTorqueById(motor_id, motor_torque)
					else:
						self._SetMotorTorqueById(motor_id, 0)
			else:
				torque_commands = -self._kp * (q - motor_commands) - self._kd * qdot

				# The torque is already in the observation space because we use
				# GetMotorAngles and GetMotorVelocities.
				self._observed_motor_torques = torque_commands

				# Transform into the motor space when applying the torque.
				self._applied_motor_torques = np.multiply(self._observed_motor_torques,
														  self._motor_direction)

				for motor_id, motor_torque in zip(self._motor_id_list, self._applied_motor_torques):
					self._SetMotorTorqueById(motor_id, motor_torque)
		else:
			motor_commands_with_direction = np.multiply(motor_commands, self._motor_direction)
			for motor_id, motor_command_with_direction in zip(self._motor_id_list,
															  motor_commands_with_direction):
				self._SetDesiredMotorAngleById(motor_id, motor_command_with_direction)

	def GetMotorAngles(self):
		"""Get the eight motor angles at the current moment.
		Returns:
		  Motor angles.
		"""
		motor_angles = [
			self._pybullet_client.getJointState(self.quadruped, motor_id)[0]
			for motor_id in self._motor_id_list
		]
		motor_angles = np.multiply(motor_angles, self._motor_direction)
		return motor_angles

	def GetMotorVelocities(self):
		"""Get the velocity of all eight motors.
		Returns:
		  Velocities of all eight motors.
		"""
		motor_velocities = [
			self._pybullet_client.getJointState(self.quadruped, motor_id)[1]
			for motor_id in self._motor_id_list
		]
		motor_velocities = np.multiply(motor_velocities, self._motor_direction)
		return motor_velocities

	def GetMotorTorques(self):
		"""Get the amount of torques the motors are exerting.
		Returns:
		  Motor torques of all eight motors.
		"""
		if self._accurate_motor_model_enabled or self._pd_control_enabled:
			return self._observed_motor_torques
		else:
			motor_torques = [
				self._pybullet_client.getJointState(self.quadruped, motor_id)[3]
				for motor_id in self._motor_id_list
			]
			motor_torques = np.multiply(motor_torques, self._motor_direction)
		return motor_torques

	def ConvertFromLegModel(self, actions):
		"""Convert the actions that use leg model to the real motor actions.
		Args:
		  actions: The theta, phi of the leg model.
		Returns:
		  The eight desired motor angles that can be used in ApplyActions().
		"""

		motor_angle = copy.deepcopy(actions)
		scale_for_singularity = 1
		offset_for_singularity = 1.5
		half_num_motors = int(self.num_motors / 2)
		quater_pi = math.pi / 4
		for i in range(self.num_motors):
			action_idx = i // 2
			forward_backward_component = (
					-scale_for_singularity * quater_pi *
					(actions[action_idx + half_num_motors] + offset_for_singularity))
			extension_component = (-1) ** i * quater_pi * actions[action_idx]
			if i >= half_num_motors:
				extension_component = -extension_component
			motor_angle[i] = (math.pi + forward_backward_component + extension_component)
		return motor_angle

	def GetBaseMassFromURDF(self):
		"""Get the mass of the base from the URDF file."""
		return self._base_mass_urdf

	def GetLegMassesFromURDF(self):
		"""Get the mass of the legs from the URDF file."""
		return self._leg_masses_urdf

	def SetBaseMass(self, base_mass):
		self._pybullet_client.changeDynamics(self.quadruped, BASE_LINK_ID, mass=base_mass)

	def SetLegMasses(self, leg_masses):
		"""Set the mass of the legs.
		A leg includes leg_link and motor. All four leg_links have the same mass,
		which is leg_masses[0]. All four motors have the same mass, which is
		leg_mass[1].
		Args:
		  leg_masses: The leg masses. leg_masses[0] is the mass of the leg link.
			leg_masses[1] is the mass of the motor.
		"""
		for link_id in LEG_LINK_ID:
			self._pybullet_client.changeDynamics(self.quadruped, link_id, mass=leg_masses[0])
		for link_id in MOTOR_LINK_ID:
			self._pybullet_client.changeDynamics(self.quadruped, link_id, mass=leg_masses[1])

	def SetFootFriction(self, foot_friction):
		"""Set the lateral friction of the feet.
		Args:
		  foot_friction: The lateral friction coefficient of the foot. This value is
			shared by all four feet.
		"""
		for link_id in FOOT_LINK_ID:
			self._pybullet_client.changeDynamics(self.quadruped, link_id, lateralFriction=foot_friction)

	def SetBatteryVoltage(self, voltage):
		if self._accurate_motor_model_enabled:
			self._motor_model.set_voltage(voltage)

	def SetMotorViscousDamping(self, viscous_damping):
		if self._accurate_motor_model_enabled:
			self._motor_model.set_viscous_damping(viscous_damping)


"""ABSTRACT BASE CLASS FOR RANDOMIZING THE ENVIRONMENT"""


class EnvRandomizerBase(object):
	"""Abstract base class for environment randomizer.
	An EnvRandomizer is called in environment.reset(). It will
	randomize physical parameters of the objects in the simulation.
	The physical parameters will be fixed for that episode and be
	randomized again in the next environment.reset().
	"""

	__metaclass__ = abc.ABCMeta

	@abc.abstractmethod
	def randomize_env(self, env):
		"""Randomize the simulated_objects in the environment.
		Args:
		  env: The environment to be randomized.
		"""
		pass


"""MINITAUR ENV RANDOMIZER"""
# Relative range.
MINITAUR_BASE_MASS_ERROR_RANGE = (-0.2, 0.2)  # 0.2 means 20%
MINITAUR_LEG_MASS_ERROR_RANGE = (-0.2, 0.2)  # 0.2 means 20%
# Absolute range.
BATTERY_VOLTAGE_RANGE = (14.8, 16.8)  # Unit: Volt
MOTOR_VISCOUS_DAMPING_RANGE = (0, 0.01)  # Unit: N*m*s/rad (torque/angular vel)
MINITAUR_LEG_FRICTION = (0.8, 1.5)  # Unit: dimensionless


class MinitaurEnvRandomizer(EnvRandomizerBase):
	"""A randomizer that change the minitaur_gym_env during every reset."""

	def __init__(self,
				 minitaur_base_mass_err_range=MINITAUR_BASE_MASS_ERROR_RANGE,
				 minitaur_leg_mass_err_range=MINITAUR_LEG_MASS_ERROR_RANGE,
				 battery_voltage_range=BATTERY_VOLTAGE_RANGE,
				 motor_viscous_damping_range=MOTOR_VISCOUS_DAMPING_RANGE):
		self._minitaur_base_mass_err_range = minitaur_base_mass_err_range
		self._minitaur_leg_mass_err_range = minitaur_leg_mass_err_range
		self._battery_voltage_range = battery_voltage_range
		self._motor_viscous_damping_range = motor_viscous_damping_range

	def randomize_env(self, env):
		self._randomize_minitaur(env.minitaur)

	def _randomize_minitaur(self, minitaur):
		"""Randomize various physical properties of minitaur.
		It randomizes the mass/inertia of the base, mass/inertia of the legs,
		friction coefficient of the feet, the battery voltage and the motor damping
		at each reset() of the environment.
		Args:
		  minitaur: the Minitaur instance in minitaur_gym_env environment.
		"""
		base_mass = minitaur.GetBaseMassFromURDF()
		randomized_base_mass = random.uniform(
			base_mass * (1.0 + self._minitaur_base_mass_err_range[0]),
			base_mass * (1.0 + self._minitaur_base_mass_err_range[1]))
		minitaur.SetBaseMass(randomized_base_mass)

		leg_masses = minitaur.GetLegMassesFromURDF()
		leg_masses_lower_bound = np.array(leg_masses) * (1.0 + self._minitaur_leg_mass_err_range[0])
		leg_masses_upper_bound = np.array(leg_masses) * (1.0 + self._minitaur_leg_mass_err_range[1])
		randomized_leg_masses = [
			np.random.uniform(leg_masses_lower_bound[i], leg_masses_upper_bound[i])
			for i in range(len(leg_masses))
		]
		minitaur.SetLegMasses(randomized_leg_masses)

		randomized_battery_voltage = random.uniform(BATTERY_VOLTAGE_RANGE[0], BATTERY_VOLTAGE_RANGE[1])
		minitaur.SetBatteryVoltage(randomized_battery_voltage)

		randomized_motor_damping = random.uniform(MOTOR_VISCOUS_DAMPING_RANGE[0],
												  MOTOR_VISCOUS_DAMPING_RANGE[1])
		minitaur.SetMotorViscousDamping(randomized_motor_damping)

		randomized_foot_friction = random.uniform(MINITAUR_LEG_FRICTION[0], MINITAUR_LEG_FRICTION[1])
		minitaur.SetFootFriction(randomized_foot_friction)


"""ACTUAL MINITAUR ENVIRONMENT"""
NUM_SUBSTEPS = 5
NUM_MOTORS = 8
MOTOR_ANGLE_OBSERVATION_INDEX = 0
MOTOR_VELOCITY_OBSERVATION_INDEX = MOTOR_ANGLE_OBSERVATION_INDEX + NUM_MOTORS
MOTOR_TORQUE_OBSERVATION_INDEX = MOTOR_VELOCITY_OBSERVATION_INDEX + NUM_MOTORS
BASE_ORIENTATION_OBSERVATION_INDEX = MOTOR_TORQUE_OBSERVATION_INDEX + NUM_MOTORS
ACTION_EPS = 0.01
OBSERVATION_EPS = 0.01
RENDER_HEIGHT = 720
RENDER_WIDTH = 960


class MinitaurBulletEnv(gym.Env):
	"""The gym environment for the minitaur.
	It simulates the locomotion of a minitaur, a quadruped robot. The state space
	include the angles, velocities and torques for all the motors and the action
	space is the desired motor angle for each motor. The reward function is based
	on how far the minitaur walks in 1000 steps and penalizes the energy
	expenditure.
	"""
	metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

	def __init__(
			self,
			urdf_root=pybullet_data.getDataPath(),
			action_repeat=1,
			distance_weight=1.0,
			energy_weight=0.005,
			shake_weight=0.0,
			drift_weight=0.0,
			distance_limit=float("inf"),
			observation_noise_stdev=0.0,
			self_collision_enabled=True,
			motor_velocity_limit=np.inf,
			pd_control_enabled=False,
			# not needed to be true if accurate motor model is enabled (has its own better PD)
			leg_model_enabled=True,
			accurate_motor_model_enabled=True,
			motor_kp=1.0,
			motor_kd=0.02,
			torque_control_enabled=False,
			motor_overheat_protection=True,
			hard_reset=True,
			on_rack=False,
			render=False,
			kd_for_pd_controllers=0.3,
			env_randomizer=MinitaurEnvRandomizer()):
		"""Initialize the minitaur gym environment.
		Args:
		  urdf_root: The path to the urdf data folder.
		  action_repeat: The number of simulation steps before actions are applied.
		  distance_weight: The weight of the distance term in the reward.
		  energy_weight: The weight of the energy term in the reward.
		  shake_weight: The weight of the vertical shakiness term in the reward.
		  drift_weight: The weight of the sideways drift term in the reward.
		  distance_limit: The maximum distance to terminate the episode.
		  observation_noise_stdev: The standard deviation of observation noise.
		  self_collision_enabled: Whether to enable self collision in the sim.
		  motor_velocity_limit: The velocity limit of each motor.
		  pd_control_enabled: Whether to use PD controller for each motor.
		  leg_model_enabled: Whether to use a leg motor to reparameterize the action
			space.
		  accurate_motor_model_enabled: Whether to use the accurate DC motor model.
		  motor_kp: proportional gain for the accurate motor model.
		  motor_kd: derivative gain for the accurate motor model.
		  torque_control_enabled: Whether to use the torque control, if set to
			False, pose control will be used.
		  motor_overheat_protection: Whether to shutdown the motor that has exerted
			large torque (OVERHEAT_SHUTDOWN_TORQUE) for an extended amount of time
			(OVERHEAT_SHUTDOWN_TIME). See ApplyAction() in minitaur.py for more
			details.
		  hard_reset: Whether to wipe the simulation and load everything when reset
			is called. If set to false, reset just place the minitaur back to start
			position and set its pose to initial configuration.
		  on_rack: Whether to place the minitaur on rack. This is only used to debug
			the walking gait. In this mode, the minitaur's base is hanged midair so
			that its walking gait is clearer to visualize.
		  render: Whether to render the simulation.
		  kd_for_pd_controllers: kd value for the pd controllers of the motors
		  env_randomizer: An EnvRandomizer to randomize the physical properties
			during reset().
		"""
		self._time_step = 0.01
		self._action_repeat = action_repeat
		self._num_bullet_solver_iterations = 300
		self._urdf_root = urdf_root
		self._self_collision_enabled = self_collision_enabled
		self._motor_velocity_limit = motor_velocity_limit
		self._observation = []
		self._env_step_counter = 0
		self._is_render = render
		self._last_base_position = [0, 0, 0]
		self._distance_weight = distance_weight
		self._energy_weight = energy_weight
		self._drift_weight = drift_weight
		self._shake_weight = shake_weight
		self._distance_limit = distance_limit
		self._observation_noise_stdev = observation_noise_stdev
		self._action_bound = 1
		self._pd_control_enabled = pd_control_enabled
		self._leg_model_enabled = leg_model_enabled
		self._accurate_motor_model_enabled = accurate_motor_model_enabled
		self._motor_kp = motor_kp
		self._motor_kd = motor_kd
		self._torque_control_enabled = torque_control_enabled
		self._motor_overheat_protection = motor_overheat_protection
		self._on_rack = on_rack
		self._cam_dist = 1.0
		self._cam_yaw = 0
		self._cam_pitch = -30
		self._hard_reset = True
		self._kd_for_pd_controllers = kd_for_pd_controllers
		self._last_frame_time = 0.0
		print("urdf_root=" + self._urdf_root)
		self._env_randomizer = env_randomizer
		# PD control needs smaller time step for stability.
		if pd_control_enabled or accurate_motor_model_enabled:
			self._time_step /= NUM_SUBSTEPS
			self._num_bullet_solver_iterations /= NUM_SUBSTEPS
			self._action_repeat *= NUM_SUBSTEPS

		if self._is_render:
			self._pybullet_client = bc.BulletClient(connection_mode=pybullet.GUI)
		else:
			self._pybullet_client = bc.BulletClient()

		self.seed()
		self.reset()
		observation_high = (self.minitaur.GetObservationUpperBound() + OBSERVATION_EPS)
		observation_low = (self.minitaur.GetObservationLowerBound() - OBSERVATION_EPS)
		action_dim = 8
		action_high = np.array([self._action_bound] * action_dim)
		self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)
		self.observation_space = spaces.Box(observation_low, observation_high, dtype=np.float32)
		self.viewer = None
		self._hard_reset = hard_reset  # This assignment need to be after reset()

	def set_env_randomizer(self, env_randomizer):
		self._env_randomizer = env_randomizer

	def configure(self, args):
		self._args = args

	def reset(self):
		if self._hard_reset:
			self._pybullet_client.resetSimulation()
			self._pybullet_client.setPhysicsEngineParameter(
				numSolverIterations=int(self._num_bullet_solver_iterations))
			self._pybullet_client.setTimeStep(self._time_step)
			plane = self._pybullet_client.loadURDF("%s/plane.urdf" % self._urdf_root)
			self._pybullet_client.changeVisualShape(plane, -1, rgbaColor=[1, 1, 1, 0.9])
			self._pybullet_client.configureDebugVisualizer(
				self._pybullet_client.COV_ENABLE_PLANAR_REFLECTION, 0)
			self._pybullet_client.setGravity(0, 0, -10)
			acc_motor = self._accurate_motor_model_enabled
			motor_protect = self._motor_overheat_protection
			self.minitaur = (Minitaur(pybullet_client=self._pybullet_client,
									  urdf_root=self._urdf_root,
									  time_step=self._time_step,
									  self_collision_enabled=self._self_collision_enabled,
									  motor_velocity_limit=self._motor_velocity_limit,
									  pd_control_enabled=self._pd_control_enabled,
									  accurate_motor_model_enabled=acc_motor,
									  motor_kp=self._motor_kp,
									  motor_kd=self._motor_kd,
									  torque_control_enabled=self._torque_control_enabled,
									  motor_overheat_protection=motor_protect,
									  on_rack=self._on_rack,
									  kd_for_pd_controllers=self._kd_for_pd_controllers))
		else:
			self.minitaur.Reset(reload_urdf=False)

		if self._env_randomizer is not None:
			self._env_randomizer.randomize_env(self)

		self._env_step_counter = 0
		self._last_base_position = [0, 0, 0]
		self._objectives = []
		self._pybullet_client.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw,
														 self._cam_pitch, [0, 0, 0])
		if not self._torque_control_enabled:
			for _ in range(100):
				if self._pd_control_enabled or self._accurate_motor_model_enabled:
					self.minitaur.ApplyAction([math.pi / 2] * 8)
				self._pybullet_client.stepSimulation()
		return self._noisy_observation()

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def _transform_action_to_motor_command(self, action):
		if self._leg_model_enabled:
			for i, action_component in enumerate(action):
				if not (-self._action_bound - ACTION_EPS <= action_component <=
						self._action_bound + ACTION_EPS):
					raise ValueError("{}th action {} out of bounds.".format(i, action_component))
			action = self.minitaur.ConvertFromLegModel(action)
		return action

	def step(self, action):
		"""Step forward the simulation, given the action.
		Args:
		  action: A list of desired motor angles for eight motors.
		Returns:
		  observations: The angles, velocities and torques of all motors.
		  reward: The reward for the current state-action pair.
		  done: Whether the episode has ended.
		  info: A dictionary that stores diagnostic information.
		Raises:
		  ValueError: The action dimension is not the same as the number of motors.
		  ValueError: The magnitude of actions is out of bounds.
		"""
		if self._is_render:
			# Sleep, otherwise the computation takes less time than real time,
			# which will make the visualization like a fast-forward video.
			time_spent = time.time() - self._last_frame_time
			self._last_frame_time = time.time()
			time_to_sleep = self._action_repeat * self._time_step - time_spent
			if time_to_sleep > 0:
				time.sleep(time_to_sleep)
			base_pos = self.minitaur.GetBasePosition()
			camInfo = self._pybullet_client.getDebugVisualizerCamera()
			curTargetPos = camInfo[11]
			distance = camInfo[10]
			yaw = camInfo[8]
			pitch = camInfo[9]
			targetPos = [
				0.95 * curTargetPos[0] + 0.05 * base_pos[0], 0.95 * curTargetPos[1] + 0.05 * base_pos[1],
				curTargetPos[2]
			]

			self._pybullet_client.resetDebugVisualizerCamera(distance, yaw, pitch, base_pos)
		action = self._transform_action_to_motor_command(action)
		for _ in range(self._action_repeat):
			self.minitaur.ApplyAction(action)
			self._pybullet_client.stepSimulation()

		self._env_step_counter += 1
		reward = self._reward()
		done = self._termination()
		return np.array(self._noisy_observation()), reward, done, {}

	def render(self, mode="rgb_array", close=False):
		if mode != "rgb_array":
			return np.array([])
		base_pos = self.minitaur.GetBasePosition()
		view_matrix = self._pybullet_client.computeViewMatrixFromYawPitchRoll(
			cameraTargetPosition=base_pos,
			distance=self._cam_dist,
			yaw=self._cam_yaw,
			pitch=self._cam_pitch,
			roll=0,
			upAxisIndex=2)
		proj_matrix = self._pybullet_client.computeProjectionMatrixFOV(fov=60,
																	   aspect=float(RENDER_WIDTH) /
																			  RENDER_HEIGHT,
																	   nearVal=0.1,
																	   farVal=100.0)
		(_, _, px, _,
		 _) = self._pybullet_client.getCameraImage(width=RENDER_WIDTH,
												   height=RENDER_HEIGHT,
												   viewMatrix=view_matrix,
												   projectionMatrix=proj_matrix,
												   renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
		rgb_array = np.array(px)
		rgb_array = rgb_array[:, :, :3]
		return rgb_array

	def get_minitaur_motor_angles(self):
		"""Get the minitaur's motor angles.
		Returns:
		  A numpy array of motor angles.
		"""
		return np.array(self._observation[MOTOR_ANGLE_OBSERVATION_INDEX:MOTOR_ANGLE_OBSERVATION_INDEX +
																		NUM_MOTORS])

	def get_minitaur_motor_velocities(self):
		"""Get the minitaur's motor velocities.
		Returns:
		  A numpy array of motor velocities.
		"""
		return np.array(
			self._observation[MOTOR_VELOCITY_OBSERVATION_INDEX:MOTOR_VELOCITY_OBSERVATION_INDEX +
															   NUM_MOTORS])

	def get_minitaur_motor_torques(self):
		"""Get the minitaur's motor torques.
		Returns:
		  A numpy array of motor torques.
		"""
		return np.array(
			self._observation[MOTOR_TORQUE_OBSERVATION_INDEX:MOTOR_TORQUE_OBSERVATION_INDEX +
															 NUM_MOTORS])

	def get_minitaur_base_orientation(self):
		"""Get the minitaur's base orientation, represented by a quaternion.
		Returns:
		  A numpy array of minitaur's orientation.
		"""
		return np.array(self._observation[BASE_ORIENTATION_OBSERVATION_INDEX:])

	def is_fallen(self):
		"""Decide whether the minitaur has fallen.
		If the up directions between the base and the world is larger (the dot
		product is smaller than 0.85) or the base is very low on the ground
		(the height is smaller than 0.13 meter), the minitaur is considered fallen.
		Returns:
		  Boolean value that indicates whether the minitaur has fallen.
		"""
		orientation = self.minitaur.GetBaseOrientation()
		rot_mat = self._pybullet_client.getMatrixFromQuaternion(orientation)
		local_up = rot_mat[6:]
		pos = self.minitaur.GetBasePosition()
		return (np.dot(np.asarray([0, 0, 1]), np.asarray(local_up)) < 0.85 or pos[2] < 0.13)

	def _termination(self):
		position = self.minitaur.GetBasePosition()
		distance = math.sqrt(position[0] ** 2 + position[1] ** 2)
		return self.is_fallen() or distance > self._distance_limit

	def _reward(self):
		current_base_position = self.minitaur.GetBasePosition()
		forward_reward = current_base_position[0] - self._last_base_position[0]
		drift_reward = -abs(current_base_position[1] - self._last_base_position[1])
		shake_reward = -abs(current_base_position[2] - self._last_base_position[2])
		self._last_base_position = current_base_position
		energy_reward = np.abs(
			np.dot(self.minitaur.GetMotorTorques(),
				   self.minitaur.GetMotorVelocities())) * self._time_step
		reward = (self._distance_weight * forward_reward - self._energy_weight * energy_reward +
				  self._drift_weight * drift_reward + self._shake_weight * shake_reward)
		self._objectives.append([forward_reward, energy_reward, drift_reward, shake_reward])
		return reward

	def get_objectives(self):
		return self._objectives

	def _get_observation(self):
		self._observation = self.minitaur.GetObservation()
		return self._observation

	def _noisy_observation(self):
		self._get_observation()
		observation = np.array(self._observation)
		if self._observation_noise_stdev > 0:
			observation += (
					np.random.normal(scale=self._observation_noise_stdev, size=observation.shape) *
					self.minitaur.GetObservationUpperBound())
		return observation

	if parse_version(gym.__version__) < parse_version('0.9.6'):
		_render = render
		_reset = reset
		_seed = seed
		_step = step
