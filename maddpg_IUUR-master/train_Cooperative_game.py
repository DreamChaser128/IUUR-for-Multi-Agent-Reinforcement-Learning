import argparse
import numpy as np
import tensorflow as tf
import time
import pickle
import os
import maddpg_IUUR.common.tf_util as U
from maddpg_IUUR.trainer.maddpg_cooperative import MADDPGAgentTrainer,MADDPGAgent_public_network
import tensorflow.contrib.layers as layers
def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="spread_10", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=10000, help="number of episodes")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.93, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    parser.add_argument("--K", type=int, default=5000, help="the learning round number for each agent")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default='Cooperative_game', help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="./tmp/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="./tmp/", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--plots-dir", type=str, default="./tmp_curves/", help="directory where plot data is saved")
    return parser.parse_args()

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def make_env(scenario_name):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios
    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

def get_trainers(env, obs_shape_n, arglist):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    for i in range(0, env.n):
        trainers.append(trainer( "agent_%d" % i,obs_shape_n, i, arglist))

    public_network = MADDPGAgent_public_network(model, obs_shape_n, env.action_space, arglist)
    return trainers,public_network

#use GPU
def make_session():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    """Returns a session that will use GPU only"""
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def train(arglist):
    with make_session():
        # Create environment
        env = make_env(arglist.scenario)
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        trainers,public_network = get_trainers(env,obs_shape_n, arglist)
        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.display or arglist.restore:
            print('Loading previous state...')
            U.load_state(arglist.load_dir,100000)

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        agent_rewards_eachstep = [[0.0] for _ in range(env.n)]
        agent_rewards_episode=0
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver()
        obs_n = env.reset()
        episode_step = 0
        train_step = [0 for _ in range(env.n)]
        train_index = [0 for _ in range(env.n)]
        t_start = time.time()
        update_step = 0
        print('Starting iterations...')
        index_agent=0
        p_index=0
        while True:
            curr_index = index_agent % 10
            # get all actions
            action_n = public_network.action(obs_n, curr_index)
            # environment step
            new_obs_n, rew_n, done_n, info_n= env.step(action_n)
            episode_step += 1
            done = all(done_n)
            terminal = (episode_step >= arglist.max_episode_len)
            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i])
            obs_n = new_obs_n

            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew
                agent_rewards_eachstep[i][-1] += rew
                agent_rewards_episode+=rew

            if terminal:
                interaction = 'steps：' + str(len(episode_rewards)) + ',episode_rewards:' + str(
                    np.mean(episode_rewards[-arglist.save_rate:]))
                print('\r{}'.format(interaction), end='')

                agent_rewards_eachstep = [[0.0] for _ in range(env.n)]
                agent_rewards_episode = 0


            if done or terminal:
                obs_n = env.reset()
                episode_step = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])

            # increment global step counter
            train_step[curr_index] =train_step[curr_index]+ 1
            train_index[p_index] += 1
            update_step += 1
            # for displaying learned policies
            if arglist.display:
                time.sleep(0.1)
                env.render()
                continue

            # update all trainers, if not in display
            trainers[curr_index].replay_sample_index = None
            loss = trainers[curr_index].update(trainers, p_index,public_network,update_step)
            if train_index[p_index]>=7000:
                train_index[p_index]=0
                p_index+=1
                p_index=p_index%10

            # save model, display training output
            if terminal and (len(episode_rewards) % arglist.save_rate == 0):
                U.save_state(arglist.save_dir,len(episode_rewards), saver=saver)
                # print statement depends on whether or not there are adversaries
                print('\n')
                print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]), round(time.time()-t_start, 3)))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                rew_file_name = str(arglist.plots_dir) + str(arglist.exp_name) + '_rewards_now_' + str(
                    len(episode_rewards)) + '.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                # Keep track of final episode reward
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > arglist.num_episodes:
                rew_file_name =str(arglist.plots_dir) + str(arglist.exp_name) + '_rewards.pkl'
                print(str(arglist.plots_dir)+',' + str(arglist.exp_name) + '_rewards.pkl')
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                agrew_file_name = str(arglist.plots_dir) + str(arglist.exp_name) + '_agent_rewards.pkl'
                print(str(arglist.plots_dir) + str(arglist.exp_name) + '_agent_rewards.pkl')
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                break

            if train_step[curr_index]>=arglist.K:
                train_step[curr_index]=0
                index_agent+=1
            else:
                 pass


if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
