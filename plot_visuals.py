import numpy as np
import pickle
import tensorflow as tf
from interact_drive.reward_design.mpc_ord import MPC_ORD
from interact_drive.reward_design.mpc_ord import finite_horizon_env
from experiments.local_opt_scenario import local_opt_env
from experiments.replanning_world import setup_world as replanning_env
# from multiprocessing import Pool
# import multiprocessing
from pathos.multiprocessing import ProcessPool as Pool
from multiprocess import context as ctx
from PIL import Image

def main():

    # save_path = f'cmaes_finite_horizon__designer_weights_[-0.07028325 0. 0. 0. -0.0843399 -0.7028325 -0.7028325 ]__5_init_seed_1335516235_opt_seed_1335516235_sigma_0.05.pkl.bak'
    save_path = f'cmaes_finite_horizon__designer_weights_[-0.07028325 0. 0. 0. -0.0843399 -0.7028325 -0.7028325 ]__5_init_seed_2035017862_opt_seed_2035017862_sigma_0.05.pkl'



    with open(save_path, 'rb') as file:
        data = pickle.load(file)

    print('Read results from \'', save_path, '\'\n\n')

    print("Tuned weights:", data[len(data)-1][0])
    # for i in range(len(data)):
    #     print('iteration ', i, ': ', data[i][0])


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

    env = envs['finite_horizon']
    # env = envs['local_opt']
    car_true, world_true, init_states = env['make_env'](debug=True)
    car_tuned, world_tuned, _ = env['make_env'](debug=True)
    print("Evaluating weights with INITIAL STATES:", init_states)

    # ## Serial processing
    # mpc_ord = MPC_ORD(world, car,
    # 				   init_states, env['eval_horizon'], num_samples=env['num_eval_samples'],
    # 				   save_path=save_path )


    # world.reset()
    # heatmap_frame = world.render(mode='rgb_array', heatmap_show=True)
    # im = Image.fromarray(heatmap_frame)
    # im.save(f"TEST__true_weights__heatmapt.png")

    # mpc_ord.eval_weights(car.weights, gif=f'TEST__true_weights.gif', heatmap_show=False)

    # car.weights = data[len(data)-1][0]
    # world.reset()
    # heatmap_frame = world.render(mode='rgb_array', heatmap_show=True)
    # im = Image.fromarray(heatmap_frame)
    # im.save(f'TEST__tuned_weights__heatmap.png')

    # mpc_ord.eval_weights(data[len(data)-1][0], gif=f'TEST__tuned_weights.gif', heatmap_show=False)


    ## Parallel processing

    tf.debugging.set_log_device_placement(True)
    gpus = tf.config.experimental.list_logical_devices('GPU')
    if gpus:
        with Pool(2) as pool:
            pool.map(draw, zip( [None, data[len(data)-1][0]],
                                ["true", "tuned"],
                                [env]*2,
                                [save_path]*2,
                                gpus
                            )
            )


def draw(_):
    weights, case, env, save_path, gpu= _ 
    print(gpu.name)
    with tf.device(gpu.name):
        car, world, init_states = env['make_env'](debug=True)
        mpc_ord = MPC_ORD(world, car,
                    init_states, env['eval_horizon'], num_samples=env['num_eval_samples'],
                    save_path=save_path )
        if case != "true":
            assert case == "tuned"
            car.weights = weights
        world.reset()
        heatmap_frame = world.render(mode='rgb_array', heatmap_show=True)
        im = Image.fromarray(heatmap_frame)
        # im.save(f"TEST__{case}_weights__heatmap.png")
        # im.save(f"TEST__local_opt__{case}_weights__heatmap.png")
        im.save(f"TEST__finite_horizon__{case}_weights__heatmap.png")

    

    # mpc_ord.eval_weights(car.weights, gif=f'TEST__{case}_weights.gif', heatmap_show=False)
    # mpc_ord.eval_weights(car.weights, gif=f'TEST__local_opt__{case}_weights.gif', heatmap_show=False)
    mpc_ord.eval_weights(car.weights, gif=f'TEST__finite_horizon__{case}_weights.gif', heatmap_show=False)


if __name__ == '__main__':
    # multiprocessing.set_start_method('spawn') 
    # multiprocessing.set_start_method('forkserver') 
    ctx._force_start_method('spawn')	# setting context for parallel 
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            total_gpu_mem = 5120
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=total_gpu_mem/2.0)]*2
            )
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)
        # try:
        #     total_gpu_mem = 5120
        #     tf.config.set_logical_device_configuration(
        #         gpus[0],
        #         [tf.config.LogicalDeviceConfiguration(memory_limit=total_gpu_mem/n_parallel)]*n_parallel
        #     )

        #     logical_devices = tf.config.list_logical_devices('GPU')
        #     assert len(logical_devices) == len(gpus) + 1
        #     # for i in range(len(logical_devices)):
        #     # 	tf.config.experimental.set_memory_growth(logical_devices[i], False)
        # except:
        #     # Invalid device or cannot modify logical devices once initialized.
        #     pass
        # logical_devices = tf.config.list_logical_devices('GPU')
        # print('\n\n****',logical_devices)
    main()
    print("\nDone.")
    input("Press ENTER to continue...")
