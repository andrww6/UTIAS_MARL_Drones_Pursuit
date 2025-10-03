import numpy as np

class PursuerAgent:
    def __init__(self, eval_net, idx, ctrl_num=1):
        self.idx = idx
        self.eval_net = eval_net  # the evaluation (core) network of agent
        self.mutated = False
        self.ctrl_num = ctrl_num
        self.name = f"agent_{idx}"
        self.train_iter = 0
        self.train_step = 0

    def learn(self, **kwargs):
        raise NotImplementedError("'learn' method should be overridden by subclasses")

    def store_transition(self, **kwargs):
        raise NotImplementedError("'store_transition' method should be overridden by subclasses")

    def choose_action(self, **kwargs):
        raise NotImplementedError("'choose_action' method should be overridden by subclasses")

    def cal_orientation_vector(self, env, choose_action, index=None, param_dic=None):
        raise NotImplementedError("'choose_action' method should be overridden by subclasses")

    def check_param_dic(self, param_dic):
        raise NotImplementedError("'check_params' method should be overridden by subclasses")

    def self_copy(self, config, idx=None):
        if config["algorithm"] in ['PPO', 'MAPPO']:
            param_temp = config["action_dim"]
        else:
            param_temp = config["num_action"]

        eval_net_copy = None
        if self.eval_net is not None:
            state_dict = self.eval_net.state_dict()
            eval_net_copy = type(self.eval_net)(param_temp)
            eval_net_copy.load_state_dict(state_dict)
            
        copy_agent = type(self)(config_args=config, idx=idx,pretrain_net=eval_net_copy)

        return copy_agent
    
    def explore_self(self, idx, agent_name, train_iter, train_step):
        config = self.config
        if config["algorithm"] in ['PPO', 'MAPPO']:
            param_temp = config["action_dim"]
        else:
            param_temp = config["num_action"]

        eval_net_copy = None
        if self.eval_net is not None:
            state_dict = self.eval_net.state_dict()
            eval_net_copy = type(self.eval_net)(param_temp)
            eval_net_copy.load_state_dict(state_dict)
            
        copy_agent = type(self)(config_args=config, idx=idx,pretrain_net=eval_net_copy)
        copy_agent.name = agent_name
        copy_agent.train_iter = train_iter
        copy_agent.train_step = train_step
        # Mutate paramters
        if np.random.random() < config["mutate_prob"]:
            self.mutated = True
            mutation = np.random.choice([0.75, 1.25])
            eps = min(
                (1 - self.gae_lambda) / 2,      # If lam is > 0.5, avoid going over 1
                self.gae_lambda / 2             # If lam is < 0.5, avoid going under 0
            )
            rnd_direction = (-1)**np.random.randint(2) 
            mutation = rnd_direction * eps
            copy_agent.gae_lambda = self.gae_lambda + mutation
            
            mutation = np.random.choice([0.75, 1.25])
            copy_agent.ppo_epoch = max(int(self.ppo_epoch * mutation), 1)
            
            mutation = np.random.choice([0.75, 1.25])
            copy_agent.lr = self.lr * mutation
            
            mutation = np.random.choice([0.75, 1.25])
            copy_agent.c1 = self.c1 * mutation
            
            mutation = np.random.choice([0.75, 1.25])
            copy_agent.c2 = self.c2 * mutation
            
            mutation = np.random.choice([0.75, 1.25])
            copy_agent.eps_clip = self.eps_clip  * mutation
            
            print("MUTATED PARAMS: gae_lambda - {} from {}, ppo_epoch - {} from {}, lr - {} from {}, c1 - {} from {}, c2 - {} from {}, eps_clip - {} from {}".format(
                copy_agent.gae_lambda, self.gae_lambda,
                copy_agent.ppo_epoch, self.ppo_epoch,
                copy_agent.lr, self.lr,
                copy_agent.c1, self.c1,
                copy_agent.c2, self.c2,
                copy_agent.eps_clip, self.eps_clip,
            ))
        return copy_agent
