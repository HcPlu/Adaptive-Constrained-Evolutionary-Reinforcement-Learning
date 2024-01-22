# -*- coding:utf-8 _*-
"""
=================================================
@Project -> File ：ES_UCB_AGV -> ucb_sampler.py
@Author ：HcPlu
@Version: 1.0
@Date: 2022/12/5 13:39
@@Description: 
==================================================
"""
import numpy as np

def mm_norm(a):
    a = np.array(a)
    return (a-np.min(a))/(np.max(a)-np.min(a))

def z_score(a):
    a = np.array(a)
    return (a-a.mean())/a.std()

def optimal_norm(a,k):
    a = np.array(a)
    a_max = np.array([-1755,-1841,-1837,-1897,-1830,-1850,-1905,-1827])
    a_min = np.ones(8)*(-2300)
    return (a-a_min[k]*np.ones(a.shape))/(a_max[k]*np.ones(a.shape)-a_min[k]*np.ones(a.shape))


def ucb_sampler_negative(instance_rewards,instance_counts,alpha=1,norm_mode = "min_max",score_func="negative"):
    ucb_list = np.zeros(8)
    total_count = np.sum(instance_counts)
    norm_rewards = []
    for k,v in instance_rewards.items():

        if norm_mode =="optimal_norm":
            norm_reward = - optimal_norm(v, k)
        elif norm_mode =="z_score":
            norm_reward = z_score(v)
        elif norm_mode =="min_max":
            norm_reward = mm_norm(v)
        else:
            norm_reward = v
        norm_rewards.append(norm_reward)

    if score_func=="negative":
        for k,norm_reward in enumerate(norm_rewards):
            count = instance_counts[k]
            ucb_list[k] = np.mean(norm_reward)+ alpha*np.sqrt(2*np.log(total_count)/count)
    elif score_func == "percentage":
        sum_score = np.sum(norm_rewards)
        for k,norm_reward in enumerate(norm_rewards):
            count = instance_counts[k]
            ucb_list[k] = np.mean(norm_reward/sum_score)+ alpha*np.sqrt(2*np.log(total_count)/count)

    return ucb_list

