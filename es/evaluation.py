import numpy as np

from joblib import delayed

CONTINUOUS_ENVS = ('LunarLanderContinuous', "MountainCarContinuous", "BipedalWalker")

def eval_policy(policy, env, instance_id=0,n_steps=200):
    total_reward = 0
    total_cost = 0
    total_steps = 0
    env.r_point=instance_id
    obs = env.reset()

    for i in range(n_steps):
        total_steps+=1
        valida_action = env.valid_actions(obs)
        # if env_name in CONTINUOUS_ENVS:
        #     action = policy.predict(np.array(env.state_phi(obs)).reshape(1, -1),valida_action, scale="tanh")
        # else:
        action = policy.predict(np.array(env.state_phi(obs)).reshape(1, -1),valida_action, scale="softmax")
        new_obs, reward, done, info = env.step(action)
        
        total_reward = total_reward + reward
        total_cost += info["cost"]
        obs = new_obs

        if done:
            break

    return total_reward,total_cost,total_steps,instance_id


def eval_policy_rule(policy, env, instance_id=0, n_steps=200,rule="fcfs"):
    total_reward = 0
    total_cost = 0
    total_steps = 0
    env.r_point = instance_id
    obs = env.reset()
    # self.policys = [fcfs, edd, nvf, std]
    rule_dict = {"fcfs":0,"edd":1,"nvf":2,"std":3}

    for i in range(n_steps):
        total_steps += 1
        valida_action = env.valid_actions(obs)
        # if env_name in CONTINUOUS_ENVS:
        #     action = policy.predict(np.array(env.state_phi(obs)).reshape(1, -1),valida_action, scale="tanh")
        # else:
        # action = policy.predict(np.array(env.state_phi(obs)).reshape(1, -1), valida_action, scale="softmax")
        action = valida_action[rule_dict[rule]]
        new_obs, reward, done, info = env.step(action)

        total_reward = total_reward + reward
        total_cost += info["cost"]
        obs = new_obs

        if done:
            break

    return total_reward, total_cost, total_steps, instance_id

# for parallel
eval_policy_delayed = delayed(eval_policy)
eval_policy_delayed_rule = delayed(eval_policy_rule)