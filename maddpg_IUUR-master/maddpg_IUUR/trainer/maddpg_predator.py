import numpy as np
import tensorflow as tf
import maddpg_IUUR.common.tf_util as U
from maddpg_IUUR.common.distributions import make_pdtype
from maddpg_IUUR import AgentTrainer
from maddpg_IUUR.trainer.replay_buffer import ReplayBuffer
import maddpg_IUUR.common.tf_util_q as U_q
import maddpg_IUUR.common.tf_util_p as U_p
import maddpg_IUUR.common.tf_util_target as U_target

def make_update_exp(vals, target_vals):
    polyak = 1.0 - 1e-2
    expression = []
    for var, var_target in zip(sorted(vals, key=lambda v: v.name), sorted(target_vals, key=lambda v: v.name)):
        expression.append(var_target.assign(polyak * var_target + (1.0-polyak) * var))
    expression = tf.group(*expression)
    return U_target.function([], [], updates=[expression])

def p_train(make_obs_ph_n, act_space_n,  p_index,p_func, q_func, optimizer, grad_norm_clipping=None, num_units=64, scope="trainer", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]
        # set up placeholders
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]
        p_input = obs_ph_n[p_index:6]
        p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="p_func", num_units=num_units)
        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))
        f = tf.reshape(p, [-1, 30])
        c = tf.split(f, 6, 1)
        act_sample1 = []
        reg_list=0
        for i in range(len(c)):
            act_pd = act_pdtype_n[i].pdfromflat(c[i])
            reg_list+=tf.reduce_mean(tf.square(act_pd.flatparam()))
            act_sample1.append(act_pd.sample())
        p_reg = reg_list/6
        act_input_n = act_ph_n + []
        q_input = tf.concat(obs_ph_n + act_input_n, 1)
        q = q_func(q_input, 1, scope="q_func", reuse=True, num_units=num_units)[:,0]
        pg_loss = -tf.reduce_mean(q)

        loss = pg_loss + p_reg * 1e-3

        optimize_expr = U.minimize_and_clip(optimizer, loss, p_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U_target.function(inputs=obs_ph_n + act_ph_n, outputs=loss, updates=[optimize_expr])
        act = U.function(inputs=obs_ph_n[p_index:6], outputs=act_sample1)
        p_values = U.function(obs_ph_n[p_index:6], p)

        # target network
        target_p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="target_p_func", num_units=num_units)
        target_f = tf.reshape(target_p, [-1, 30])
        c_target = tf.split(target_f, 6, 1)
        target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))
        update_target_p = make_update_exp(p_func_vars, target_p_func_vars)
        act_sample2 = []
        for i in range(len(c_target)):
            target_act_sample = act_pdtype_n[i].pdfromflat(c_target[i])
            act_sample2.append(target_act_sample.sample())

        target_act = U_p.function(inputs=obs_ph_n[p_index:6], outputs=act_sample2)
        return act, train, update_target_p, {'p_values': p_values, 'target_act': target_act}



def q_train(make_obs_ph_n, act_space_n, q_func, optimizer, grad_norm_clipping=None, scope="trainer", reuse=None, num_units=64):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]
        # set up placeholders
        obs_ph_n = make_obs_ph_n

        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]
        target_ph = tf.placeholder(tf.float32, [None], name="target")
        q_input = tf.concat(obs_ph_n + act_ph_n, 1)

        q = q_func(q_input, 1, scope="q_func", num_units=num_units)[:,0]
        q_func_vars = U_q.scope_vars(U_q.absolute_scope_name("q_func"))

        q_loss = tf.reduce_mean(tf.square(q - target_ph))

        # viscosity solution to Bellman differential equation in place of an initial condition
        q_reg = tf.reduce_mean(tf.square(q))
        loss = q_loss #+ 1e-3 * q_reg

        optimize_expr = U_q.minimize_and_clip(optimizer, loss, q_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U_q.function(inputs=obs_ph_n + act_ph_n + [target_ph], outputs=loss, updates=[optimize_expr])

        q_values = U_q.function(obs_ph_n + act_ph_n, q)

        # target network
        target_q = q_func(q_input, 1, scope="target_q_func", num_units=num_units)[:,0]
        target_q_func_vars = U_q.scope_vars(U_q.absolute_scope_name("target_q_func"))
        update_target_q = make_update_exp(q_func_vars, target_q_func_vars)
        target_q_values = U_q.function(obs_ph_n + act_ph_n, target_q)

        return train, update_target_q, {'q_values': q_values, 'target_q_values': target_q_values}


class MADDPGAgent_public_network_predator():
    def __init__(self,  model,obs_shape_n, act_space_n, args):
        obs_ph_n_q = []
        obs_ph_n_p = []
        self.n = len(obs_shape_n)
        self.agent_index=0
        for i in range(self.n):
            obs_ph_n_q.append(U.BatchInput(obs_shape_n[i], name="observation"+str(i)).get())
        obs_ph_n_q.append(U.BatchInput((26,), name="observation" + str(self.n)).get())


        for i in range(self.n):
            obs_ph_n_p.append(U.BatchInput(obs_shape_n[i], name="observation" + str(i)).get())
        obs_ph_n_p.append(U.BatchInput((26,), name="observation" + str(self.n)).get())
        # Create all the functions necessary to train the model
        self.q_train, self.q_update, self.q_debug = q_train(
            make_obs_ph_n=obs_ph_n_q,
            act_space_n=act_space_n,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            num_units=args.num_units,
            scope="trainer_preys"
        )

        self.act, self.p_train, self.p_update, self.p_debug = p_train(
            make_obs_ph_n=obs_ph_n_p,
            act_space_n=act_space_n,
            p_index=self.agent_index,
            p_func=model,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            num_units=args.num_units,
            scope="trainer_preys"
        )


    def action(self, obs,p_index):
        self.agent_index=p_index
        acts1=[]
        obss=[]
        for i in range(len(obs)):
            obss.append([obs[i]])
        acts =self.act(obss[:6])
        for i in range(len(acts)):
            acts1.append(acts[i][0])
        return acts1

class MADDPGAgentTrainer_predator(AgentTrainer):

    def __init__(self, name, obs_shape_n, agent_index, args):
        self.name = name
        self.n = len(obs_shape_n)
        self.agent_index = agent_index
        self.args = args
        # Create experience buffer
        self.replay_buffer = ReplayBuffer(5000)
        self.max_replay_buffer_len = 5000  # 容量：2500
        self.replay_sample_index = None

    def experience(self, obs, act, rew, new_obs, done):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, rew, new_obs, float(done))

    def preupdate(self):
        self.replay_sample_index = None

    def update(self, agents,p_index,public_network_prey,public_network_good,t):
        if len(self.replay_buffer) < self.max_replay_buffer_len:
            return
        if not t % 100 == 0:  # only update every 100 steps
            return
        self.replay_sample_index = self.replay_buffer.make_index(self.args.batch_size)
        # collect replay sample from all agents
        obs_n = []
        obs_next_n = []
        act_n = []
        index = self.replay_sample_index

        for i in range(self.n):
            obs, act, rew, obs_next, done = agents[i].replay_buffer.sample_index(index)
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)
        obs, act, rew, obs_next, done = self.replay_buffer.sample_index(index)
        # train q network
        num_sample = 1
        target_q = 0.0
        if self.agent_index ==p_index:
            target_act_next_n =  public_network_prey.p_debug['target_act'](obs_next_n[:(self.n - 2)])
            target_act_next_ns=target_act_next_n[0]
            target_act_next_nn=public_network_good.p_debug['target_act'](obs_next_n[6:self.n])
            target_act_next_ns.append(target_act_next_nn[0][0])
            target_act_next_ns.append(target_act_next_nn[0][1])
            obs_next_nc = obs_next_n.copy()
            obs_next_nc.append(obs_next)
            target_q_next = public_network_prey.q_debug['target_q_values'](*(obs_next_nc + target_act_next_ns))
            target_q += rew + self.args.gamma * (1.0 - done) * target_q_next
        else:
            target_act_next_n = public_network_prey.p_debug['target_act'](obs_next_n[:(self.n - 2)])
            target_act_next_ns = target_act_next_n[0]
            target_act_next_nn = public_network_good.p_debug['target_act'](obs_next_n[6:self.n])
            target_act_next_ns.append(target_act_next_nn[0][0])
            target_act_next_ns.append(target_act_next_nn[0][1])
            obs_next_nc = obs_next_n.copy()
            obs_next_nc.append(obs_next)
            target_q_next = public_network_prey.q_debug['target_q_values'](*(obs_next_nc + target_act_next_ns))
            target_q += target_q_next

        target_q /= num_sample
        obs_nc = obs_n.copy()
        obs_nc.append(obs)
        q_loss = public_network_prey.q_train(*(obs_nc + act_n + [target_q]))

        # train p network
        p_loss = public_network_prey.p_train(*(obs_nc + act_n))
        #update network
        public_network_prey.p_update()
        public_network_prey.q_update()

        return [q_loss, p_loss, np.mean(target_q), np.mean(rew), np.mean(target_q_next), np.std(target_q)]

