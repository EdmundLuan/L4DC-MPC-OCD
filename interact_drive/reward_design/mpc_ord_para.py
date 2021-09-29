from multiprocessing import spawn
from experiments.merging import ThreeLaneCarWorld, ThreeLaneTestCar
from interact_drive.car import FixedVelocityCar
from pathos.multiprocessing import ProcessPool as Pool
from multiprocess import context as ctx
# from multiprocessing import Pool
import multiprocessing
import numpy as np
import tensorflow as tf
import time
import pickle
import cma
import scipy.stats


class list2(list): # hack to make a mutable list type in python
	def __init__(self, *args, **kwargs):
		super().__init__(self, *args, **kwargs)

class MPC_ORD:
	def __init__(self, world, car, init_car_states, designer_horizon, batch_sz=1, save_path=None, num_samples=1):
		self.batch_sz = batch_sz if batch_sz>= 1 else 1
		self.worlds = [world]*self.batch_sz
		self.cars = [car]*self.batch_sz

		self.designer_horizon = designer_horizon
		self.init_car_states = init_car_states
		self.save_path = save_path

		self.designer_weights = car.weights/np.linalg.norm(car.weights)
		self.weight_dim = len(car.weights)

		self.zero_time = time.time()
		self.history = list2()
		self.iter = 0
		self.should_save_history = False
		self.done = False
		self.num_samples = num_samples

	def optimize_cmaes(self, seed=1, sigma0=0.1):
		self.history.seed = seed
		assert seed != 0
		assert not self.done
		self.should_save_history = True

		self.eval_weights(self.designer_weights)
		
		print('Start optimizing...')

		x, es = cma.evolution_strategy.fmin2(self.eval_weights, list(self.designer_weights), sigma0=sigma0, options={'seed': seed})

		self.should_save_history = False
		self.done = True
		print('########## Done. ############')
		return x

	def optimize_random_search(self, n_iter=1000, seed=1):
		self.history.seed = seed
		assert not self.done
		self.should_save_history = True
		print("\n\nSTARTING RANDOM SEARCH")
		self.iter = 0


		# print designer results for reference
		self.eval_weights(self.designer_weights)

		np.random.seed(seed)

		for _ in range(n_iter):
			weights = np.random.rand(*self.designer_weights.shape)*2-1
			self.eval_weights(weights)
		self.should_save_history = False
		self.done = True
		return max(self.history, key=lambda a: a[1])

	def eval_weights_for_init(self, _):
		# evaluates weights for a single initialization
		
		world, car, init, weights, render, heatmap_show, gpu = _
		with tf.device(gpu.name):
			if weights.ndim ==  2:
				weights = weights[0] # reshape 2d array with one row into 1d array
			weights = weights/np.linalg.norm(weights)
			if car.debug:
				for car_ in world.cars:
					assert car_.debug

			frames = []
			def maybe_render():
				if render:
					frames.append(self.world.render("rgb_array", heatmap_show=heatmap_show))
			
			car.weights = weights
			car.init_state = tf.constant(init, dtype=tf.float32)

			designer_reward = 0
			for _ in range(self.num_samples):

				world.reset()
				#if not self.car.debug:
				maybe_render()

				sample_reward = 0

				for i in range(self.designer_horizon):
					past_state, controls, state = world.step()
					#if not car.debug:
					maybe_render()
					sample_reward += car.reward_fn(past_state, controls[car.index], weights=self.designer_weights)
				#if car.debug:
				#	maybe_render()
				designer_reward += sample_reward
				print('sample_reward',sample_reward)
			designer_reward = designer_reward.numpy()
			print('init', init, '\nweights', weights ,'\n\tgave return:',designer_reward, '\ntime',time.time()-self.zero_time)
		return (designer_reward, frames) if render else designer_reward


	def eval_weights(self, weights, gif=None, heatmap_show=False):
		if gif:
			import contextlib
			with contextlib.redirect_stdout(None):  # disable the pygame import print
				from moviepy.editor import ImageSequenceClip

		if isinstance(weights, list):
			weights = np.array(weights)

		if weights.ndim == 2:
			weights = weights[0] # reshape 2d array with one row into 1d array
		weights = weights/np.linalg.norm(weights)
		
		print('------------------------------------------')
		print('ITERATION', self.iter)
		print('eval',weights)


		total_designer_reward = 0
		frames = []

		## Parallel processing, but doesn't work
		# n_parallel = 4
		# n_parallel = multiprocessing.cpu_count()
		# n_parallel = self.n_parallel
		# gpus = tf.config.experimental.list_physical_devices('GPU')
		# if gpus:
		# 	# Restrict TensorFlow to only allocate 1GB of memory on the first GPU
		# 	try:
		# 		total_gpu_mem = 5120
		# 		tf.config.experimental.set_virtual_device_configuration(
		# 			gpus[0],
		# 			[
		# 				tf.config.experimental.VirtualDeviceConfiguration(memory_limit=total_gpu_mem/self.batch_sz) 
		# 			]*self.batch_sz
		# 		)
		# 		logical_gpus = tf.config.experimental.list_logical_devices('GPU')
		# 		print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
		# 	except RuntimeError as e:
		# 		# Virtual devices must be set before GPUs have been initialized
		# 		print(e)
		# input('Holding...')
		tf.debugging.set_log_device_placement(True)
		gpus = tf.config.experimental.list_logical_devices('GPU')
		with Pool(self.batch_sz) as pool:
			print('Evaluating ', self.batch_sz, ' trajectories in parallel...')
			# print(self.worlds,
			# 	self.cars,
			# 	self.init_car_states, 
			# 	[weights]*self.batch_sz,
			# 	[gif is not None]*self.batch_sz, 
			# 	[heatmap_show]*self.batch_sz)
			ret = pool.map(self.eval_weights_for_init , 
						zip(
							self.worlds,
							self.cars,
							self.init_car_states, 
							[weights]*self.batch_sz,
							[gif is not None]*self.batch_sz, 
							[heatmap_show]*self.batch_sz, 
							gpus
						)
					)
		if gif:
			rewrds, frames = ret
		else:
			rewrds = ret
		if self.batch_sz>1:
			print('\t rewards from init states: ', rewrds)
		
		# total_designer_reward += sum(rewrds)

		# Serial processing
		# for init_car_state in self.init_car_states:

		# 	r = self.eval_weights_for_init(init_car_state, weights, gif is not None, heatmap_show=heatmap_show)
		# 	if gif:
		# 		r, frames_from_init = r
		# 		frames.extend(frames_from_init)

		# 	if len(self.init_car_states) > 1:
		# 		print('\tinit_reward',r)
		# 	total_designer_reward += r

		total_designer_reward /= self.num_samples
		print('eval reward for weights:',total_designer_reward,'\n\n')

		if gif:
			clip = ImageSequenceClip(frames, fps=int(1 / self.world.dt))
			clip.speedx(0.5).write_gif(gif, program="ffmpeg")

		self.history.append((weights, total_designer_reward))
		if self.should_save_history and self.save_path is not None:
			self.save_history()
		self.iter += 1

		return -total_designer_reward  # minus since we're minimizing the cost

	def save_history(self):
		assert self.save_path is not None
		with open(self.save_path, 'wb') as file:
			pickle.dump(self.history, file)

		print('Wrote results so far to',self.save_path, '\n\n')



def finite_horizon_env(horizon=5, env_seeds=[1], debug=True, extra_inits=False):
	"""
	Runs a planning car in a merging scenario for 15 steps and visualizes it.
	The weights of our planning car should cause it to merge into the right lane
	between the two other cars.
	"""
	ROBOT_X_MEAN, ROBOT_X_STD, ROBOT_X_RANGE = 0, 0.04, (-0.1, 0.1) # clip gaussian at bounds
	ROBOT_Y_MEAN, ROBOT_Y_STD, ROBOT_Y_RANGE = -0.9, 0.02, (-0.95, -0.85) # clip gaussian at bounds
	ROBOT_SPEED_MEAN, ROBOT_SPEED_STD, ROBOT_SPEED_RANGE = 0.8, 0.03, (0.7, 0.9)
	def get_init_state(env_seed):
		rng = np.random.RandomState(env_seed)
		np.random.seed(seed=env_seed)
		def sample(mean, std, rang):
			a, b = (rang[0] - mean) / std, (rang[1] - mean) / std
			return np.squeeze(scipy.stats.truncnorm.rvs(a,b)*std+mean)

		robot_x = sample(ROBOT_X_MEAN, ROBOT_X_STD, ROBOT_X_RANGE)
		robot_y = sample(ROBOT_Y_MEAN, ROBOT_Y_STD, ROBOT_Y_RANGE)
		robot_init_speed = sample(ROBOT_SPEED_MEAN, ROBOT_SPEED_STD, ROBOT_SPEED_RANGE)
		return np.array([robot_x, robot_y, robot_init_speed, np.pi / 2])

	init_states = [get_init_state(s) for s in env_seeds]
	world = ThreeLaneCarWorld(visualizer_args=dict(name="Switch Lanes"))

	our_car = ThreeLaneTestCar(
		world,
		init_state=init_states[0], #,np.array([0, -0.5, 0.8, np.pi / 2]),
		horizon=horizon,
		weights=np.array([-5, 0., 0., 0., -6., -50, -50]),
		debug=debug,
		planner_args=dict(n_iter=200 if horizon==6 else 100, extra_inits=extra_inits)
	)
	other_car= FixedVelocityCar(
		world,
		np.array([0, -0.6, 0.5, np.pi / 2]),
		horizon=horizon,
		color="gray",
		opacity=0.8,
		debug=debug,
		planner_args=dict(n_iter=200 if horizon==6 else 100, extra_inits=extra_inits)
	)
	world.add_cars([our_car, other_car])

	world.reset()

	return our_car, world, init_states