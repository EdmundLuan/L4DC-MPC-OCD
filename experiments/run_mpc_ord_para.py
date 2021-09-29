from argparse import ArgumentParser
from pathos.multiprocessing import ProcessPool as Pool
from multiprocess import context as ctx
from PIL.Image import init
from interact_drive.reward_design.mpc_ord_para import finite_horizon_env
from experiments.local_opt_scenario import local_opt_env
from experiments.replanning_world import setup_world as replanning_env
import numpy as np
import tensorflow as tf
from interact_drive.reward_design.mpc_ord_para import MPC_ORD


def fmt(arr):
	s = str(arr).replace("\n", ' ').replace('\t', " ")
	while '  ' in s:
		s = s.replace('  ', ' ')
	return s

envs = {
	'local_opt': {
		'make_env': local_opt_env,
		'eval_horizon': 15,
		'init_offset_range': [[0.,-0.1,0.,0.], [0.,0.1,0.,0.]],
		'num_eval_samples': 1,
		'tuned_weights': np.array([-0.09686739,  0.25720383, -0.58355971, -0.23075428, -0.41237239,
       -0.4758984 , -0.36625558])
	}, 

	'finite_horizon': {
		'make_env': finite_horizon_env,
		'eval_horizon': 15,
		'init_offset_range': [[-0.1,0.,0.,0.], [0.1,0.,0.,0.]],
		'num_eval_samples': 1,
		'tuned_weights': np.array([-0.21963165, -0.01184596,  0.34379187, -0.04687411, -0.06364365,
       -0.54138792, -0.7308079 ])
	}, 
	'replanning': {
		'make_env': replanning_env,
		'eval_horizon': 20,
		'init_offset_range': [[-0.05,0.,0.,0.], [0.05,0.,0.,0.]],
		'num_eval_samples': 2, # need to sample each possible outcome to compute expected reward
		'tuned_weights': np.array([-0.55899817 ,-0.4436692, -0.3724511 ,-0.19964276, -0.5438697, 0.12770043])
	}, 
}
def main(n_parallel):
	parser = ArgumentParser()
	parser.add_argument('scenario', type=str, choices=['local_opt', 'finite_horizon', 'replanning'],
	                    help='Which scenario to run reward weight optimization for.')
	parser.add_argument('optimizer', type=str, choices=['random', 'cmaes', 'vis'], #, 'bayesopt'],
						help='Algorithm used for weight optimization.')

	parser.add_argument('--n_inits', type=int, default=1) # n_inits is assumed to be less than 1e6
	parser.add_argument('--seed', type=int, default=None)# --root_init_seed
	# The seeds used for robot init state are (seed*1e6 + 1), (seed*1e6 + 2), ..., (seed*1e6 + n_inits)
	# Seed used for CMAES/random search is randomly selected in advance

	parser.add_argument('--one_by_one', action='store_true', help='Runs single init optimization separately in parallel for each init.')
	parser.add_argument('--rand_inits', action='store_true')
	parser.add_argument('--sigma', type=float, default=0.05)


	args = parser.parse_args()

	assert args.n_inits >= 1
	assert args.seed != 0, 'CMA doesn\'t accept 0 seed'

	if args.n_inits == 1:
		args.one_by_one = False # no special handling



	env = envs[args.scenario]

	# use main seed for experiment to randomly sample seeds for inits
	#  # if the root seed is the same across multiple calls, the env_seeds will match;
	#  if n_inits is incremented without changing root seed, it simply adds an envseed [args.seed*1e6+(n_inits-1)] to the set of env seeds
	optimization_seed = np.random.randint(0, 2**31)
	if args.seed is None:
		args.seed = optimization_seed
	env_seeds = [(args.seed*1000000+i)%(2**32) for i in range(args.n_inits)]
	car, world, init_states = env['make_env'](env_seeds=env_seeds, debug=True)

	init_states_groups = [[s] for s in init_states] if args.one_by_one else [init_states]

	# if len(init_states_groups) > 1:
	# 	n_parallel = len(init_states_groups)
	# 	with Pool(n_parallel) as pool:
	# 		print('\n\nRunning ', n_parallel, ' tasks in parallel... ')
	# 		pool.map(run_opt, zip([env]*len(init_states_groups), 
	# 			init_states_groups, 
	# 			[args]*len(init_states_groups), 
	# 			[optimization_seed for i in range(len(init_states_groups))]
	# 		))
	# else:
	run_opt((env, init_states, args, optimization_seed, n_parallel))


def run_opt(_):
	env, init_states, args, optimization_seed, batch_sz= _

	print("\n*****************\n\tOPTIMIZING REWARD FROM INIT STATES", init_states, '\n*****************')

	car, world, _ = env['make_env'](debug=True)

	mpc_ord = MPC_ORD(world, car,
				   init_states, env['eval_horizon'], batch_sz, num_samples=env['num_eval_samples'],
				   save_path=f'{args.optimizer}_{args.scenario}__designer_weights_{fmt(car.weights)}__{args.n_inits if not args.one_by_one else fmt(init_states[0])}_init_seed_{args.seed}_opt_seed_{optimization_seed}_sigma_{args.sigma}.pkl', )

	# if args.optimizer == 'bayesopt':
	# 	mpc_ord.optimize(n_iter=400, seed=optimization_seed)
	if args.optimizer == 'random':
		mpc_ord.optimize_random_search(n_iter=400, seed=optimization_seed)
	elif args.optimizer == 'vis':
		from PIL import Image
		world.reset()
		heatmap_frame = world.render(mode='rgb_array', heatmap_show=True)
		im = Image.fromarray(heatmap_frame)
		im.save(f"{args.scenario}__true_weights_{fmt(car.weights)}__heatmap.png")
		mpc_ord.eval_weights(car.weights, gif=f'{args.scenario}__true_weights_{fmt(car.weights)}.gif', heatmap_show=False)
		car.weights = env["tuned_weights"]
		world.reset()
		heatmap_frame = world.render(mode='rgb_array', heatmap_show=True)
		im = Image.fromarray(heatmap_frame)
		im.save(f'{args.scenario}__tuned_weights_{env["tuned_weights"]}__heatmap.png')
		mpc_ord.eval_weights(env["tuned_weights"], gif=f'{args.scenario}_tuned_weights_{env["tuned_weights"]}.gif', heatmap_show=False)
	else:
		assert args.optimizer == 'cmaes'
		mpc_ord.optimize_cmaes(sigma0=args.sigma, seed=optimization_seed)


if __name__ == '__main__':
	# gpus = tf.config.experimental.list_physical_devices('GPU')
	n_parallel = eval(input("n_parallel = "))
	if n_parallel <= 0:
		n_parallel = 1
	# if gpus:
	# 	# Restrict TensorFlow to only allocate 1GB of memory on the first GPU
	# 	try:
	# 		total_gpu_mem = 5120
	# 		tf.config.experimental.set_virtual_device_configuration(
	# 			gpus[0],
	# 			[
	# 				tf.config.experimental.VirtualDeviceConfiguration(memory_limit=total_gpu_mem/n_parallel) 
	# 			]*n_parallel
	# 		)
	# 		logical_gpus = tf.config.experimental.list_logical_devices('GPU')
	# 		print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
	# 	except RuntimeError as e:
	# 		# Virtual devices must be set before GPUs have been initialized
	# 		print(e)
	gpus = tf.config.list_physical_devices('GPU')
	try:
		total_gpu_mem = 5120
		tf.config.set_logical_device_configuration(
			gpus[0],
			[tf.config.LogicalDeviceConfiguration(memory_limit=total_gpu_mem/n_parallel)]*n_parallel
		)

		logical_devices = tf.config.list_logical_devices('GPU')
		assert len(logical_devices) == len(gpus) + 1
		# for i in range(len(logical_devices)):
		# 	tf.config.experimental.set_memory_growth(logical_devices[i], False)
	except:
		# Invalid device or cannot modify logical devices once initialized.
		pass
	gpus = tf.config.list_logical_devices('GPU')
	print('\n\n****',gpus)

	input('GPU set, holding. Press Enter to continue... ')

	ctx._force_start_method('spawn')	# setting context for parallel 
	main(n_parallel)
