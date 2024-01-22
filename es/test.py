# -*- coding:utf-8 _*-
"""
=================================================
@Project -> File ：ES_UCB_AGV -> test.py
@Author ：HcPlu
@Version: 1.0
@Date: 2022/12/12 11:02
@@Description: 
==================================================
"""
from es.evaluation import eval_policy_delayed, eval_policy
import numpy as np
from es.utils.util import handle_mix_result
from joblib import Parallel
import json

def test_episode(policy,test_env,args, test_num = 240):
    file_path = args.log_path
    rewards_jobs = (eval_policy_delayed(policy, test_env, instance_id=i % 8, n_steps=args.env_steps) for i in
                    range(test_num))
    res = Parallel(n_jobs=args.jobs)(rewards_jobs)
    test_result = np.array(res)
    split_result,_ = handle_mix_result(test_result)
    print(split_result)
    save_res = []
    for k in range(8):
        instance_res = np.array(split_result[k])
        makespan = instance_res[:,0]
        tardiness = instance_res[:,1]
        save_res.append([makespan.tolist(),tardiness.tolist()])
    print(save_res)
    with open(file_path+"/test.json","w") as f:
        json.dump(save_res,f)

    return save_res