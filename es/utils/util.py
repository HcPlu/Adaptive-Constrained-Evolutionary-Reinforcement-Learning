import numpy as np

def add_train_log(logger,train_result,step,epoch):
    split_result,_ = handle_mix_result(train_result)
    for k, v in split_result.items():
        v = np.array(v)
        rewards = v[:, 0]
        costs = v[:, 1]
        steps = v[:, 2]
        logger.write("train", step, {
            "train/%d_rewards" % k: np.mean(rewards),
            "train/%d_rewards_std" % k: np.std(rewards),
            "train/%d_costs" % k: np.mean(costs),
            "train/%d_costs_std" % k: np.std(costs),
        })
    return split_result
def add_test_log(logger,test_result,step,epoch):
    split_result,_ = handle_mix_result(test_result)
    for k,v in split_result.items():
        v = np.array(v)
        rewards = v[:,0]
        costs = v[:,1]
        steps = v[:,2]
        logger.write("test",step,{
            "test/%d_rewards"%k: np.mean(rewards),
            "test/%d_rewards_std"%k: np.std(rewards),
            "test/%d_costs"%k: np.mean(costs),
            "test/%d_costs_std"%k: np.std(costs),
        })


def fixed_cam_range():
    return 2

def handle_mix_result(result):
    instance_ids = result[:,3]
    keys = np.unique(instance_ids)
    split_result = {}
    split_result_idx = {}
    for key in keys:
        split_result[int(key)] = []
        split_result_idx[int(key)] = []
    for idx,single_result in enumerate(result):
        key = int(single_result[3])
        split_result[key].append(single_result)
        split_result_idx[key].append(idx)
    return split_result,split_result_idx


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

def record_counts(logger,counts,step):
    for idx, item in enumerate(counts):
        logger.write("instance_count",step,{
            "instance/%d_count"%idx: item,

        })
