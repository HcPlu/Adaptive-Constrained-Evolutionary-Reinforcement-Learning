import numpy as np

from copy import deepcopy
from es.utils.util import handle_mix_result
import math
class OpenAiES:
    def __init__(self, model, learning_rate, noise_std, \
                    noise_decay=1.0, lr_decay=1.0, decay_step=50, norm_rewards=True,sr = False,rank=False):
        self.model = model
        
        self._lr = learning_rate
        self._noise_std = noise_std
        
        self.noise_decay = noise_decay
        self.lr_decay = lr_decay
        self.decay_step = decay_step
        self.norm_rewards = norm_rewards
        self.sr = sr
        self.rank = rank


        self._population = None
        self._count = 0

    @property
    def noise_std(self):
        step_decay = np.power(self.noise_decay, np.floor((1 + self._count) / self.decay_step))

        return self._noise_std * step_decay

    @property
    def lr(self):
        step_decay = np.power(self.lr_decay, np.floor((1 + self._count) / self.decay_step))

        return self._lr * step_decay

    def generate_population(self, npop=50):
        self._population = []

        for i in range(npop):
            new_model = deepcopy(self.model)
            new_model.E = []

            for i, layer in enumerate(new_model.W):
                noise = np.random.randn(layer.shape[0], layer.shape[1])

                new_model.E.append(noise)
                new_model.W[i] = new_model.W[i] + self.noise_std * noise
            self._population.append(new_model)

        return self._population

    def transformed_phi(self, cs, rcpo_alpha):
        return np.sum(max(np.array(cs) - np.ones(np.array(cs).shape) * rcpo_alpha, 0) ** 2)

    def stochastic_ranking(self, rewards, constraints, rcpo_alpha, pf=0.5):
        N = len(rewards)
        index_I = np.arange(N)
        for i in range(N):
            for j in range(N - 1):
                u = np.random.rand()
                front = index_I[j]
                back = index_I[j + 1]
                if (self.transformed_phi(constraints[front], rcpo_alpha) == 0 and self.transformed_phi(
                        constraints[back], rcpo_alpha) == 0) or u < pf:
                    # if abs(constraints[front]-constraints[back])<0.0000001 or u < pf:
                    # maximum problem
                    if (rewards[front] < rewards[back]):
                        index_I[j] = back
                        index_I[j + 1] = front
                else:
                    if self.transformed_phi(constraints[front], rcpo_alpha) > self.transformed_phi(constraints[back],
                                                                                                   rcpo_alpha):
                        index_I[j] = back
                        index_I[j + 1] = front
            # print(front,back,index_I)
            # print(j,j+1,rewards[index_I].tolist())

        return index_I

    def fitness_shaping(self,returns):
        """
        A rank transformation on the rewards, which reduces the chances
        of falling into local optima early in training.
        """
        sorted_returns_backwards = sorted(returns)[::-1]
        lamb = len(returns)
        shaped_returns = []
        denom = sum([max(0, math.log(lamb / 2 + 1, 2) -
                         math.log(sorted_returns_backwards.index(r) + 1, 2))
                     for r in returns])
        for r in returns:
            num = max(0, math.log(lamb / 2 + 1, 2) -
                      math.log(sorted_returns_backwards.index(r) + 1, 2))
            shaped_returns.append(num / denom + 1 / lamb)
        return shaped_returns

    def update_population(self, train_result):
        if self._population is None:
            raise ValueError("populations is none, generate & eval it first")

        # rewards = train_result[:,0]
        # costs = train_result[:,1]
        split_result,split_result_idx = handle_mix_result(train_result)
        shape_result = {}
        rewards = np.zeros(len(self._population))
        for k,single_result in split_result.items():
            single_idx = np.array(split_result_idx[k])
            single_rewards = np.array(single_result)[:, 0]
            costs = np.array(single_result)[:, 1]
        # z-normalization (?) - works better, but slower
            if self.rank:
                if self.norm_rewards:

                    # rewards = self.fitness_shaping(rewards)
                    rewards_idx = np.argsort(single_rewards)
                    score = np.zeros(single_rewards.shape)
                    score[rewards_idx] = np.arange(single_rewards.shape[0])
                    shape_rewards = (score - score.mean()) / (score.std() + 1e-5)

                if self.sr:
                    sr_index = self.stochastic_ranking(single_rewards,costs,50)
                    score = np.zeros(single_rewards.shape)
                    score[sr_index] = np.arange(single_rewards.shape[0])[::-1]
                    shape_rewards = (score - score.mean()) / (score.std() + 1e-5)
            else:
                # simply use fitness
                if self.norm_rewards:
                    # rewards = self.fitness_shaping(rewards)
                    # rewards_idx = np.argsort(single_rewards)
                    # score = np.zeros(single_rewards.shape)
                    score = single_rewards
                    shape_rewards = (score - score.mean()) / (score.std() + 1e-5)

                if self.sr:
                    # sr_index = self.stochastic_ranking(single_rewards, costs, 50)
                    # score = np.zeros(single_rewards.shape)
                    score = single_rewards - costs

                    shape_rewards = (score - score.mean()) / (score.std() + 1e-5)

            rewards[single_idx] = shape_rewards
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        for i, layer in enumerate(self.model.W):
            w_updates = np.zeros_like(layer)

            for j, model in enumerate(self._population):
                w_updates = w_updates + (model.E[i] * rewards[j])

            # SGD weights update
            self.model.W[i] = self.model.W[i] + (self.lr / (len(rewards) * self.noise_std)) * w_updates
        
        self._count = self._count + 1

    def get_model(self):
        return self.model


class OpenAIES_NSR:
    # TODO: novelity search
    def __init__(self):
        pass