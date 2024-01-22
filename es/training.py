import gym
import pickle
import uuid
from .utils import TensorboardLogger
import numpy as np
import os,datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from joblib import Parallel
from collections import defaultdict
from gym import wrappers
from wrappers.scenario_wrapper import LearnFixStructuralObsWrapper,LearnTurnStructuralObsWrapper
from es.linear import ThreeLayerNetwork
from es.es_strategy import OpenAiES
from es.plot import plot_rewards
from es.evaluation import eval_policy_delayed, eval_policy
from es.utils.util import *
from es.ucb_sampler import *
# env: (n_states, n_actions)

def save_model(policy,path,session):
    with open(path+"/check_point_%d.pkl"%session, "wb") as file:
        pickle.dump(policy, file)

def train_loop(policy, train_env, test_env, args,logger, n_jobs=1, verbose=True):
    es = OpenAiES(
        model=policy, 
        learning_rate=args.learning_rate,
        noise_std=args.noise_std,
        noise_decay=args.noise_decay,
        lr_decay=args.lr_decay,
        decay_step=args.decay_step,
        sr=args.sr,
        rank=args.rank
    )

    norm_mode = "min_max"
    score_func = "negative"
    log = defaultdict(list)
    steps = 0
    pop_size = args.population_size
    # pbar = tqdm(range(args.n_sessions))
    instance_counts = np.zeros(8)
    instance_rewards = {i:np.array([]) for i in range(8)}
    for session in range(args.n_sessions):

        # best policy stats
        if args.mode=="ucb":
            if session < 2:
                train_instance_list = np.random.randint(0, 8, pop_size)
            else:
                ucb_list = ucb_sampler_negative(instance_rewards,instance_counts,norm_mode = norm_mode, score_func = score_func)
                train_instance_list = np.random.choice(np.arange(8),pop_size,p=np.exp(ucb_list)/np.sum(np.exp(ucb_list)))
                print(ucb_list.tolist())
        elif args.mode=="random":
            train_instance_list = np.random.randint(0, 8, pop_size)
        elif args.mode=="per":
            train_instance_list = np.arange(pop_size)%8
        elif args.mode=="single":
            train_instance_list = np.ones(pop_size,dtype=np.int32)*args.train_instance
        else:
            train_instance_list = np.arange(pop_size) % 8


        for instance in train_instance_list:
            instance_counts[instance]+=1

        if session % args.save_step == 0:
            save_model(es.get_model(),args.log_path,session)

        if session % args.eval_step == 0:
            best_policy = es.get_model()
            rewards_jobs = (eval_policy_delayed(best_policy, test_env,instance_id=i%8,n_steps= args.env_steps) for i in range(args.test_num))
            res = Parallel(n_jobs=n_jobs)(rewards_jobs)
            test_result = np.array(res)
            add_test_log(logger,test_result,steps,session)

            test_rewards = test_result[:,0]
            test_costs = test_result[:, 1]


            if verbose:
                print(f"Test {session+1}/{args.n_sessions}: reward/cost: {round(np.mean(test_rewards), 4)}/{round(np.mean(test_costs), 4)}",
                      f"std: {round(np.std(test_rewards), 3)}/{round(np.std(test_costs), 3)}")

        population = es.generate_population(args.population_size)
        rewards_jobs = (eval_policy_delayed(new_policy, train_env, instance_id=train_instance_list[i], n_steps=args.env_steps) for i,new_policy in enumerate(population))
        res = Parallel(n_jobs=n_jobs)(rewards_jobs)
        train_result = np.array(res)
        rewards = train_result[:,0]
        costs = train_result[:, 1]
        steps += np.sum(train_result[:,2])
        es.update_population(train_result)

        split_train_result = add_train_log(logger,train_result,steps,session)
        for k, v in split_train_result.items():
            v = np.array(v)
            split_rewards = v[:, 0]
            split_costs = v[:, 1]
            instance_rewards[k] = np.append(instance_rewards[k],split_rewards)
        # pbar.set_description(f"epoch :{session+1}/{args.n_sessions}")
        record_counts(logger,instance_counts,steps)

        print({"session":session,"steps":steps, "lr":es.lr,"noise_std":es.noise_std,"reward":round(np.mean(rewards), 4),"cost":round(np.mean(costs), 4)})
        # populations stats

    save_model(es.get_model(),args.log_path,args.n_sessions)

    return log


def run_experiment(args,):
    n_jobs = args.jobs
    verbose = args.verbose

    scenario_name = "scenario2"
    max_task_num = 30
    max_per_time = 30
    init_tasks_num = 5
    if args.train_instance!=-1:
        train_env = LearnFixStructuralObsWrapper(gym.make(args.env),r_point=args.train_instance, max_task_num=max_task_num, max_per_time=max_per_time,
                                              init_tasks_num=init_tasks_num, scenario=scenario_name)
    else:
        train_env = LearnTurnStructuralObsWrapper(gym.make(args.env), max_task_num=max_task_num, max_per_time=max_per_time,
                                              init_tasks_num=init_tasks_num, scenario=scenario_name)
    test_env = LearnFixStructuralObsWrapper(gym.make(args.env),r_point=args.train_instance, max_task_num=max_task_num, max_per_time=max_per_time,
                                              init_tasks_num=init_tasks_num, scenario=scenario_name)
    np.random.seed(args.seed)
    train_env.seed(args.seed)

    train_env._env_name = args.env

    n_states = train_env.observation_space.shape or train_env.observation_space.n
    n_states = n_states[0]
    n_actions = train_env.action_space.shape or train_env.action_space.n
    print("obs/act",n_states,n_actions)

    # timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_path = os.path.join(args.logdir, args.env, 'es',args.experiment_name)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer)
    args.log_path = log_path


    policy = ThreeLayerNetwork(
        in_features=n_states,
        out_features=n_actions,
        hidden_sizes=args.hidden_sizes
    )
    # TODO: save model on KeyboardInterrupt exception
    log = train_loop(policy, train_env,test_env, args,logger, n_jobs, verbose)

    from es.test import test_episode

    res = test_episode(policy,test_env,args)
    res = np.array(res)
    print(np.mean(res,axis=2))
    return policy


def render_policy(model_path, env_name, n_videos=1):
    with open(model_path, "rb") as file:
        policy = pickle.load(file)

    model_name = model_path.split("/")[-1].split(".")[0]
    
    for i in range(n_videos):
        env = gym.make(env_name)
        env = wrappers.Monitor(env, f'videos/{model_name}/' + str(uuid.uuid4()), force=True)

        print(eval_policy(policy, env, n_steps=1600))
        env.close()


