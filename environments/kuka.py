"""
For the Kuka Robotic Gripper environment, provided by PyBullet.
https://github.com/bulletphysics/bullet3/blob/6ad7e8fa6e86dda0ad78eafe8902a6b1ac54254a/examples/pybullet/gym/pybullet_envs/bullet/kukaCamGymEnv.py
Wrapper for Gym-style environment provided by Euginio from Micron Systems

All code has been modified to fit the specifications of the codebase it currently resides within.

author: trevor m
"""

import inspect
import math
import os
import pybullet as p
import time
import random
import gym
import glob
import numpy as np
from numpy.random import uniform
import pybullet_data
from gym import spaces
from gym.utils import seeding
from pkg_resources import parse_version

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, current_dir)
large_val_observation = 100

RENDER_HEIGHT = 720
RENDER_WIDTH = 960


class Kuka:
	"""The class specifically to define the Kuka robot and its movement characteristics.
		this class uses flies from the pybullet_data package. These files describe the 3D building blocks of Kuka.

		The Kuka class is wrapped by KukaEnv class (defined below).

		Notes:
			μ(s) is deterministic
	"""

	def __init__(self, urdf_rootpath=pybullet_data.getDataPath(), timestep=0.01):
		self.urdf_rootpath = urdf_rootpath
		self.timestep = timestep
		self.max_velocity = .35
		self.max_force = 200.
		self.finger_a_force = 2
		self.finger_b_force = 2.5
		self.finger_tip_force = 2
		self.use_inverse_kinematics = 1
		self.use_simulation = 1
		self.use_null_space = 21
		self.use_orientation = 1
		self.kuka_end_effector_idx = 6
		self.kuka_gripper_idx = 7

		# Limits
		# lower limits for null space
		self.ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]

		# upper limits for null space
		self.ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]

		# joint ranges for null space
		self.jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]

		# restposes for null space
		self.rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]

		# joint damping coefficents
		self.jd = [
			0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001,
			0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001
		]

		self.kuka_uid = None
		self.joint_positions = None
		self.num_joints = None
		self.tray_uid = None
		self.end_effector_pos = None
		self.end_effector_angle = None
		self.motor_names = None
		self.motor_indices = None

		self.reset()

	def reset(self):
		"""Resets the robot back to a deterministic starting position."""
		objects = p.loadSDF(os.path.join(self.urdf_rootpath, "kuka_iiwa/kuka_with_gripper2.sdf"))
		self.kuka_uid = objects[0]
		p.resetBasePositionAndOrientation(self.kuka_uid,
										  [-0.100000, 0.000000, 0.070000],
										  [0.000000, 0.000000, 0.000000, 1.000000])
		self.joint_positions = [
			0.006418, 0.413184, -0.011401, -1.589317, 0.005379, 1.137684, -0.006539,
			0.000048, -0.299912, 0.000000, -0.000043, 0.299960, 0.000000, -0.000200
		]

		self.num_joints = p.getNumJoints(self.kuka_uid)

		for joint_index in range(self.num_joints):
			p.resetJointState(self.kuka_uid, joint_index, self.joint_positions[joint_index])
			p.setJointMotorControl2(self.kuka_uid, joint_index, p.POSITION_CONTROL,
									targetPosition=self.joint_positions[joint_index], force=self.max_force)

		self.tray_uid = p.loadURDF(os.path.join(self.urdf_rootpath, "tray/tray.urdf"), 0.640000, 0.075000, -0.190000,
								   0.000000, 0.000000, 1.000000, 0.000000)
		self.end_effector_pos = [0.537, 0.0, 0.5]
		self.end_effector_angle = 0

		self.motor_names = []
		self.motor_indices = []

		for i in range(self.num_joints):
			joint_info = p.getJointInfo(self.kuka_uid, i)
			q_idx = joint_info[3]
			if q_idx > -1:
				self.motor_names.append(str(joint_info[1]))
				self.motor_indices.append(i)

	def get_action_dimension(self):
		"""If not using inverse kinematics, default to 6 for:
			x, y, z, roll, pitch, yaw -- Euler angles of end affector
		"""
		if self.use_inverse_kinematics:
			return len(self.motor_indices)
		return 6

	def get_observation_dimension(self):
		return len(self.get_observation())

	def get_observation(self):
		observation = []
		state = p.getLinkState(self.kuka_uid, self.kuka_gripper_idx)
		pos = state[0]
		orn = state[1]
		euler = p.getEulerFromQuaternion(orn)

		observation.extend(list(pos))
		observation.extend(list(euler))

		return observation

	def apply_action(self, motor_commands):
		if self.use_inverse_kinematics:
			dx = motor_commands[0]
			dy = motor_commands[1]
			dz = motor_commands[2]
			da = motor_commands[3]
			filter_angle = motor_commands[4]

			state = p.getLinkState(self.kuka_uid, self.kuka_end_effector_idx)
			actualEndEffectorPos = state[0]

			self.end_effector_pos[0] = self.end_effector_pos[0] + dx
			if self.end_effector_pos[0] > 0.65:
				self.end_effector_pos[0] = 0.65

			if self.end_effector_pos[0] < 0.50:
				self.end_effector_pos[0] = 0.50

			self.end_effector_pos[1] = self.end_effector_pos[1] + dy

			if self.end_effector_pos[1] < -0.17:
				self.end_effector_pos[1] = -0.17

			if self.end_effector_pos[1] > 0.22:
				self.end_effector_pos[1] = 0.22

			self.end_effector_pos[2] = self.end_effector_pos[2] + dz

			self.end_effector_angle = self.end_effector_angle + da

			pos = self.end_effector_pos

			orn = p.getQuaternionFromEuler([0, -math.pi, 0])  # -math.pi,yaw])

			if self.use_null_space == 1:
				if self.use_orientation == 1:
					joint_poses = p.calculateInverseKinematics(self.kuka_uid, self.kuka_end_effector_idx, pos, orn,
															   self.ll, self.ul, self.jr, self.rp)
				else:
					joint_poses = p.calculateInverseKinematics(self.kuka_uid, self.kuka_end_effector_idx, pos,
															   lowerLimits=self.ll, upperLimits=self.ul,
															   jointRanges=self.jr, restPoses=self.rp)
			else:
				if self.use_orientation == 1:
					joint_poses = p.calculateInverseKinematics(self.kuka_uid, self.kuka_end_effector_idx, pos, orn,
															   jointDamping=self.jd)
				else:
					joint_poses = p.calculateInverseKinematics(self.kuka_uid, self.kuka_end_effector_idx, pos)

			if self.use_simulation:
				for i in range(self.kuka_end_effector_idx + 1):
					p.setJointMotorControl2(bodyUniqueId=self.kuka_uid, jointIndex=i, controlMode=p.POSITION_CONTROL,
											targetPosition=joint_poses[i], targetVelocity=0, force=self.max_force,
											maxVelocity=self.max_velocity, positionGain=0.3, velocityGain=1)
			else:
				for i in range(self.num_joints):
					p.resetJointState(self.kuka_uid, i, joint_poses[i])

			# fingers
			p.setJointMotorControl2(self.kuka_uid, 7, p.POSITION_CONTROL, targetPosition=self.end_effector_angle,
									force=self.max_force)
			p.setJointMotorControl2(self.kuka_uid, 8, p.POSITION_CONTROL, targetPosition=-filter_angle,
									force=self.finger_a_force)
			p.setJointMotorControl2(self.kuka_uid, 11, p.POSITION_CONTROL, targetPosition=filter_angle,
									force=self.finger_b_force)
			p.setJointMotorControl2(self.kuka_uid, 10, p.POSITION_CONTROL, targetPosition=0,
									force=self.finger_tip_force)
			p.setJointMotorControl2(self.kuka_uid, 13, p.POSITION_CONTROL, targetPosition=0,
									force=self.finger_tip_force)

		else:
			for action in range(len(motor_commands)):
				motor = self.motor_indices[action]
				p.setJointMotorControl2(self.kuka_uid, motor, p.POSITION_CONTROL, targetPosition=motor_commands[action],
										force=self.max_force)


class KukaEnv(gym.Env):
	"""The wrapper for the Kuka robot

		static (bool): static table object?
	"""

	def __init__(self, urdf_root=pybullet_data.getDataPath(), action_repeat=1, is_enable_self_collision=True,
				 renders=False, is_discrete=False, max_steps=1000, images=False, width=224, height=224,
				 static_all=True, static_obj_rnd_pos=False, rnd_obj_rnd_pos=False, full_color=True):
		self.is_discrete = is_discrete
		self.timestep = 1. / 240.
		self.urdf_root = urdf_root
		self.action_repeat = action_repeat
		self.is_enable_self_collision = is_enable_self_collision
		self.observation = []
		self.env_step_counter = 0
		self.renders = renders
		self.max_steps = max_steps
		self.terminated = 0
		self.cam_dist = 1.3
		self.cam_yaw = 180
		self.cam_pitch = -40
		self.images = images
		self.width = width
		self.height = height
		self.static_all = static_all
		self.static_obj_rnd_pos = static_obj_rnd_pos
		self.rnd_obj_rnd_pos = rnd_obj_rnd_pos
		self.full_color = full_color

		self.action_space = 7

		self.rnd_obj = None

		self.metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

		self.view_mat = [
			-0.5120397806167603, 0.7171027660369873, -0.47284144163131714, 0.0, -0.8589617609977722,
			-0.42747554183006287, 0.28186774253845215, 0.0, 0.0, 0.5504802465438843,
			0.8348482847213745, 0.0, 0.1925382763147354, -0.24935829639434814, -0.4401884973049164, 1.0
		]


		self.proj_matrix = [
			0.75, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0,
			-0.02000020071864128, 0.0
		]

		# self.view_mat = p.computeViewMatrix(
		# 	cameraEyePosition=[0.72, -0.2, 50],
		# 	cameraTargetPosition=[0.72, 0.05, -0.33],
		# 	cameraUpVector=[0, 1, 0]
		# )
		#
		# self.proj_matrix = p.computeProjectionMatrixFOV(
		# 	fov=0.7,
		# 	aspect=1.0,
		# 	nearVal=0.1,
		# 	farVal=100
		# )

		self._p = p

		if self.renders:
			cid = p.connect(p.SHARED_MEMORY)
			if cid < 0:
				cid = p.connect(p.GUI)

			p.resetDebugVisualizerCamera(1.3, 180, -41, [0.52, -0.2, -0.33])

		else:
			p.connect(p.DIRECT)

		self.seed()

		self.reset()

		observation_dim = len(self.get_extended_observation())

		observation_high = np.array([large_val_observation] * observation_dim)

		if self.is_discrete:
			self.action_space = spaces.Discrete(7)

		else:
			action_dim = 3
			self._action_bound = 1
			action_high = np.array([self._action_bound] * action_dim)
			self.action_space = spaces.Box(-action_high, action_high)

		self.observation_space = spaces.Box(-observation_high, observation_high)

		self.viewer = None
		self.block_uid = None
		self.robot = None

	def __del__(self):
		p.disconnect()

	def reset(self):
		"""Resets the environment
			Random positioning with the original block.urdf:
				x : [0.4, 0.8]
				y : [0.2, 0.3]
				z : [0.15] -- should be set here so the object can fall into the tray, otherwise may spawn underneath
		"""
		self.terminated = 0
		p.resetSimulation()
		p.setPhysicsEngineParameter(numSolverIterations=150)
		p.setTimeStep(self.timestep)
		p.loadURDF(os.path.join(self.urdf_root, "plane.urdf"), [0, 0, -1])

		p.loadURDF(os.path.join(self.urdf_root, "table/table.urdf"),
				   0.5000000, 0.00000, -.820000, 0.000000, 0.000000, 0.0, 1.0)

		# The orientation of the block via quaternions
		orn = p.getQuaternionFromEuler([0, 0, 3.14 / 2])

		drop_height = -0.1

		# Placing the block on the table at [0.55, 0, -0.15] <x,y,z>
		if self.static_all:
			self.block_uid = p.loadURDF(os.path.join(self.urdf_root, "block.urdf"),
										[0.55, 0, drop_height], orn, globalScaling=2.0)
			for _ in range(500):
				p.stepSimulation()

		elif self.static_obj_rnd_pos:
			xpos = uniform(0.5, 0.7)
			ypos = uniform(-0.1, 0.2)
			angle = np.pi / 2 + 0.3 * np.pi * random.random()
			orn = p.getQuaternionFromEuler([0, 0, angle])
			# scale = np.random.choice([x for x in range(5)])
			self.block_uid = p.loadURDF(os.path.join(self.urdf_root, "block.urdf"),
										[xpos, ypos, drop_height],
										orn,
										globalScaling=2.0)
			# Need to advance the simulation several steps. Sometimes, the objects spawn above the table
			# The original pybullet code uses 500 steps. Not sure if this many is necessary
			for _ in range(500):
				p.stepSimulation()

		elif self.rnd_obj_rnd_pos:
			urdf_list = self._get_random_object()
			for urdf_name in urdf_list:
				xpos = uniform(0.45, 0.65)
				ypos = uniform(-0.05, 0.2)
				angle = np.pi / 2 + 0.3 * np.pi * random.random()
				orn = p.getQuaternionFromEuler([0, 0, angle])
				scale = np.random.choice([x for x in range(3)])

				urdf_path = os.path.join(self.urdf_root, urdf_name)
				self.block_uid = p.loadURDF(urdf_path,
											[xpos, ypos, drop_height],
											[orn[0], orn[1], orn[2], orn[3]],
											globalScaling=1)

				for _ in range(500):
					p.stepSimulation()

		else:
			urdf_list = self._get_random_object()
			self.rnd_obj = urdf_list
			for urdf_name in urdf_list:
				orn = p.getQuaternionFromEuler([0, 0, 3.14 / 2])
				scale = np.random.choice([x for x in range(3)])
				urdf_path = os.path.join(self.urdf_root, urdf_name)
				self.block_uid = p.loadURDF(urdf_path,
											[0.55, 0, drop_height],
											orn,
											globalScaling=1)

				for _ in range(500):
					p.stepSimulation()

		p.setGravity(0, 0, -10)
		self.robot = Kuka(urdf_rootpath=self.urdf_root, timestep=self.timestep)
		self.env_step_counter = 0
		p.stepSimulation()

		if not self.images:
			self.observation = self.get_extended_observation()
			return np.array(self.observation)
		else:
			return self.get_image()

	def get_image(self):
		"""Returns an image state"""
		if self.full_color:
			self.observation = self._p.getCameraImage(height=self.height, width=self.width,
													  projectionMatrix=self.proj_matrix,
													  viewMatrix=self.view_mat)[2][:, :, :3] / 255.
		else:
			full_color = self._p.getCameraImage(height=self.height, width=self.width,
												projectionMatrix=self.proj_matrix,
												viewMatrix=self.view_mat)[2][:, :, :3] / 255.
			self.observation = (full_color[:, :, 0] + full_color[:, :, 1] + full_color[:, :, 2]) / 3
		return self.observation

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def get_extended_observation(self):
		self.observation = self.robot.get_observation()
		gripper_state = p.getLinkState(self.robot.kuka_uid, self.robot.kuka_gripper_idx)
		gripper_pos = gripper_state[0]
		gripper_orn = gripper_state[1]
		block_pos, block_orn = p.getBasePositionAndOrientation(self.block_uid)

		inv_gripper_pos, inv_gripper_orn = p.invertTransform(gripper_pos, gripper_orn)
		gripper_mat = p.getMatrixFromQuaternion(gripper_orn)
		dir0 = [gripper_mat[0], gripper_mat[3], gripper_mat[6]]
		dir1 = [gripper_mat[1], gripper_mat[4], gripper_mat[7]]
		dir2 = [gripper_mat[2], gripper_mat[5], gripper_mat[8]]

		gripper_eul = p.getEulerFromQuaternion(gripper_orn)
		# print("gripperEul")
		# print(gripperEul)
		block_pos_in_gripper, block_orn_in_gripper = p.multiplyTransforms(inv_gripper_pos, inv_gripper_orn,
																		  block_pos, block_orn)
		projected_block_pos2D = [block_pos_in_gripper[0], block_pos_in_gripper[1]]
		block_euler_in_gripper = p.getEulerFromQuaternion(block_orn_in_gripper)

		# we return the relative x,y position and euler angle of block in gripper space
		block_in_gripper_pos_xyeulz = [block_pos_in_gripper[0], block_pos_in_gripper[1], block_euler_in_gripper[2]]

		self.observation.extend(list(block_in_gripper_pos_xyeulz))

		return self.observation

	def step(self, action):
		"""Leaving all of the commented-out lines for my future reference. May come in handy. It did

			with -0.002 for Z, gets to -6.x -> -3.x reward trend. Not enough to reach object?
		"""
		if self.is_discrete:
			dv = 0.005
			# dx = [0, -dv, dv, 0, 0, 0, 0][action]
			# dy = [0, 0, 0, -dv, dv, 0, 0][action]
			# da = [0, 0, 0, 0, 0, -0.05, 0.05][action]
			# f = 0.3
			# real_action = [dx, dy, -0.002, da, f]

			# EC: limits movement space to just x,y,z
			# 0 = no action
			dx = [0, -dv, dv, 0, 0, 0, 0][action]  # x: 1,2
			dy = [0, 0, 0, -dv, dv, 0, 0][action]  # y: 3,4
			dz = [0, 0, 0, 0, 0, -dv, dv][action]  # z: 5 down,6
			da = 0  # da = [0,   0,   0,   0,   0,   0,  0, -0.05, 0.05][action] # gripper rotation
			# da = [0, 0, 0, 0, 0, 0, 0, -0.05, 0.05][action]
			f = 0.3  # gripper angle open for grasping

			# This action set makes the arm always go down
			# real_action = [dx, dy, -0.002, da, f]

			# Switched this around! With the below actions, gripper rotation/angle is constant
			real_action = [dx, dy, dz, da, f]
			# real_action = [dx, 0, dz, da, f] # original action

		else:
			dv = 0.005
			dx = action[0] * dv
			dy = action[1] * dv
			da = action[2] * 0.05
			f = 0.3
			real_action = [dx, dy, -0.002, da, f]

		# This portion used to be separated out into two different functions
		for i in range(self.action_repeat):
			self.robot.apply_action(real_action)
			p.stepSimulation()

			if self._termination():
				break

			self.env_step_counter += 1

		if self.renders:
			time.sleep(self.timestep)

		if not self.images:
			self.observation = self.get_extended_observation()

		else:
			self.observation = self.get_image()

		done = self._termination()

		# only penalize rotation until learning works well [real_action[0],real_action[1],real_action[3]])
		npaction = np.array([
			real_action[3]
		])

		action_cost = np.linalg.norm(npaction) * 10.
		reward, picked_up = self._reward() - action_cost

		return np.array(self.observation), reward, picked_up, done, {}

	def step2(self, action):
		for i in range(self.action_repeat):
			self.robot.apply_action(action)
			p.stepSimulation()
			if self._termination():
				break
			self.env_step_counter += 1
		if self.renders:
			time.sleep(self.timestep)

		if not self.images:
			self.observation = self.get_extended_observation()
		else:
			self.observation = self.get_image()
		# print("self.env_step_counter")
		# print(self.env_step_counter)

		done = self._termination()
		npaction = np.array([
			action[3]
		])  # only penalize rotation until learning works well [action[0],action[1],action[3]])
		action_cost = np.linalg.norm(npaction) * 10.
		# print("action_cost")
		# print(action_cost)
		reward, picked_up = self._reward() - action_cost
		# print("reward")
		# print(reward)

		# print("len=%r" % len(self.observation))

		return np.array(self.observation), reward, picked_up, done, {}

	def render(self, mode="rgb_array", close=False):
		if mode != "rgb_array":
			return np.array([])

		base_pos, orn = self._p.getBasePositionAndOrientation(self.robot.kuka_uid)
		# view_matrix = self._p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=base_pos,
		# 														distance=self.cam_dist,
		# 														yaw=self.cam_yaw,
		# 														pitch=self.cam_pitch,
		# 														roll=0,
		# 														upAxisIndex=2)
		view_matrix = self._p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=base_pos,
																distance=self.cam_dist,
																yaw=self.cam_yaw,
																pitch=self.cam_pitch,
																roll=0,
																upAxisIndex=2)
		proj_matrix = self._p.computeProjectionMatrixFOV(fov=60,
														 aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
														 nearVal=0.1,
														 farVal=100.0)
		(_, _, px, _, _) = self._p.getCameraImage(width=RENDER_WIDTH,
												  height=RENDER_HEIGHT,
												  viewMatrix=view_matrix,
												  projectionMatrix=proj_matrix,
												  renderer=self._p.ER_BULLET_HARDWARE_OPENGL)
		# renderer=self._p.ER_TINY_RENDERER)

		rgb_array = np.array(px, dtype=np.uint8)
		rgb_array = np.reshape(rgb_array, (RENDER_HEIGHT, RENDER_WIDTH, 4))

		rgb_array = rgb_array[:, :, :3]
		return rgb_array

	def _termination(self):
		state = p.getLinkState(self.robot.kuka_uid, self.robot.kuka_end_effector_idx)
		actualEndEffectorPos = state[0]

		# print("self.env_step_counter")
		# print(self.env_step_counter)
		if self.terminated or self.env_step_counter > self.max_steps:
			if not self.images:
				self.observation = self.get_extended_observation()
			else:
				self.observation = self.get_image()
			return True

		max_dist = 0.005
		closest_points = p.getClosestPoints(self.robot.tray_uid, self.robot.kuka_uid, max_dist)

		# old note -- (actualEndEffectorPos[2] <= -0.43):
		if len(closest_points):
			# start grasp and terminate
			filter_angle = 0.3

			for i in range(100):
				grasp_action = [0, 0, 0.0001, 0, filter_angle]
				self.robot.apply_action(grasp_action)
				p.stepSimulation()
				filter_angle = filter_angle - (0.3 / 100.)

				if filter_angle < 0:
					filter_angle = 0

			for i in range(1000):
				grasp_action = [0, 0, 0.001, 0, filter_angle]
				self.robot.apply_action(grasp_action)
				p.stepSimulation()
				block_pos, block_orn = p.getBasePositionAndOrientation(self.block_uid)

				if block_pos[2] > 0.23:
					break

				state = p.getLinkState(self.robot.kuka_uid, self.robot.kuka_end_effector_idx)
				actualEndEffectorPos = state[0]

				if actualEndEffectorPos[2] > 0.5:
					break

			if not self.images:
				self.observation = self.get_extended_observation()

			else:
				self.observation = self.get_image()

		return False

	def _get_random_object(self, test=0, num_objects=1):
		"""

		Args:
			test (bool):
			num_objects (int): this can be adjusted to drop multiple objects in the tray

		Returns:

		"""
		passed_objs = [
			'random_urdfs/359/359.urdf', 'random_urdfs/935/935.urdf', 'random_urdfs/199/199.urdf',
			'random_urdfs/978/978.urdf', 'random_urdfs/182/182.urdf'
		]
		if test:
			urdf_pattern = os.path.join(self.urdf_root, 'random_urdfs/*0/*.urdf')
		else:
			obj = np.random.choice(passed_objs)
			# urdf_pattern = os.path.join(self.urdf_root, 'random_urdfs/*[1-9]/*.urdf')
			urdf_pattern = os.path.join(self.urdf_root, obj)

		found_object_directories = glob.glob(urdf_pattern)
		total_num_objects = len(found_object_directories)
		selected_objects = np.random.choice(np.arange(total_num_objects), num_objects)
		selected_objects_filenames = []
		for object_index in selected_objects:
			selected_objects_filenames += [found_object_directories[object_index]]

		return selected_objects_filenames

	def _reward(self):

		picked_up = 0

		# rewards is height of target object
		block_pos, block_orn = p.getBasePositionAndOrientation(self.block_uid)

		# Computes the distance between base (-1) of self.block_uid and the
		# self.kuka_end_effector_idx component of self.robot.kuka_uid
		# Any objects outside of 1000 units not counted
		# Returns list of closest points
		closest_points = p.getClosestPoints(self.block_uid, self.robot.kuka_uid, 1000, -1,
											self.robot.kuka_end_effector_idx)

		reward = -1000

		num_pt = len(closest_points)

		if num_pt > 0:
			reward = -closest_points[0][8] * 10

		# Could we make the task easier by lowering the height condition?
		# 3rd item from block_pos vector is the block's Z-coordinate
		if block_pos[2] > 0.2:
			picked_up = 1
			reward = reward + 10000  # 10000
			print("Successfully grasped a block!")

		return reward, picked_up

	if parse_version(gym.__version__) < parse_version('0.9.6'):
		_render = render
		_reset = reset
		_seed = seed
		_step = step
