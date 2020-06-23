import numpy as np
import math
import random
import matplotlib.pyplot as plt
import time
from scipy.stats import bernoulli

def generate_arm_reward(n_arm, each_arm_mean):
    each_arm_reward = []
    for i in xrange(n_arm):
        each_arm_reward.append(bernoulli.rvs(each_arm_mean[i],size=1))
    return each_arm_reward

def cal_kl_divergence(p,q):
    v = p*math.log(p/float(q))+(1-p)*math.log((1-p)/float(1-q))
    return v

# policy
def take_action(attempt, policy, n_arm, each_arm_empirical_reward_sum, each_arm_count, previous_action_num, each_arm_reward):
    if policy == 1: # KL-UCB
        if attempt <= n_arm:
            action_num = attempt % n_arm
        else:
            f_t = 1+attempt*math.log(attempt)*math.log(attempt)
            right = math.log(f_t)/each_arm_count
            empirical_mean = each_arm_empirical_reward_sum / each_arm_count
            save_mu = np.zeros(shape=(n_arm,))
            for i in xrange(n_arm):
                epsilon = 1e-5
                mu_til = 1
                if empirical_mean[i]>=1:
                    empirical_mean[i]=1-1e-12
                if empirical_mean[i]<=0:
                    empirical_mean[i] = 0 + 1e-12
                if mu_til >= 1:
                    mu_til = 1-1e-12
                if mu_til<= 0:
                    mu_til = 0+1e-12
                while 1:
                    if cal_kl_divergence(empirical_mean[i], mu_til)<=right[i]:
                        break

                    else:
                        mu_til -= epsilon

                save_mu[i] = mu_til

            action_num = np.argmax(save_mu)

    if policy == 2: # UCB
        if attempt <= n_arm:
            action_num = attempt % n_arm
        else:
            delta = 1 / float(attempt*attempt)
            empirical_mean = each_arm_empirical_reward_sum / each_arm_count
            confidence_width = []
            for i in xrange(n_arm):
                confidence_width.append(math.sqrt((2*math.log(1/delta))/each_arm_count[i]))
            UCB = empirical_mean + confidence_width
            action_num = np.argmax(UCB)

    return action_num, previous_action_num

def cal_regret(attempt, each_arm_count, each_arm_mean):
    best_arm = np.argmax(each_arm_mean)
    best = (attempt) * each_arm_mean[best_arm]

    empirical_reward_sum = 0
    for i in xrange(len(each_arm_count)):
        empirical_reward_sum += each_arm_count[i]*each_arm_mean[i]

    regret = best - empirical_reward_sum

    return regret

def cal_regret_upperbound(policy, n, k, each_arm_mean):
    # KL-UCB
    if policy == 1:
        upperbound = 0
    # UCB
    if policy == 2:
        # Theorem 7.1 #############################################
        # best = max(each_arm_mean)
        # gap = []
        # for i in xrange(k):
        #     gap.append(best - each_arm_mean[i])
        # first_term = sum(gap)
        # second_term = 0
        # for i in xrange(k):
        #     if gap[i] > 0:
        #         second_term += 16*math.log(n)/float(gap[i])
        # upperbound = 3*first_term + second_term
        ###########################################################

        # Theorem 7.2 #############################################
        best = max(each_arm_mean)
        gap = []
        for i in xrange(k):
            gap.append(best - each_arm_mean[i])
        first_term = 8*math.sqrt(n*k*math.log(n))
        second_term = 3*sum(gap)
        upperbound = first_term + second_term
        ###########################################################

    return upperbound

total_attempt_in_each_experience = 1000
total_experience_num = 100

# arm info
n_arm = 10
each_arm_mean = np.random.uniform(low=0, high=1, size=(n_arm,))

policy = [1, 2]
policy_name = ['KL-UCB', 'UCB']

compare_regret = np.zeros(shape=(2, total_attempt_in_each_experience + 1))

for policy_num in xrange(len(policy)):

    regret_n_upperbound = np.zeros(shape=(2, total_attempt_in_each_experience + 1))
    tmp_regret_save = np.zeros(shape=(total_experience_num, total_attempt_in_each_experience + 1))
    tmp_y_save = np.zeros(shape=(total_experience_num, n_arm, total_attempt_in_each_experience/n_arm+1))

    for experience_num in xrange(total_experience_num):
        print ("experience num : %d" % experience_num)
        each_arm_count = np.zeros(shape=(n_arm,), dtype=np.float64)
        each_arm_empirical_reward_sum = np.zeros(shape=(n_arm,), dtype=np.float64)
        to_find_max_sum_reward = np.zeros(shape=(n_arm,), dtype=np.float64)

        previous_action_num = 0

        y = np.zeros(shape=(n_arm,total_attempt_in_each_experience/n_arm+1))

        for attempt in xrange(1, total_attempt_in_each_experience+1):
            each_arm_reward = generate_arm_reward(n_arm, each_arm_mean)

            action_num, previous_action_num = take_action(attempt, policy[policy_num], n_arm, each_arm_empirical_reward_sum, each_arm_count, previous_action_num, each_arm_reward)

            each_arm_count[action_num] += 1
            each_arm_empirical_reward_sum[action_num] += each_arm_reward[action_num]


            if (experience_num + 1) % 1 == 0:
                if attempt % 100 == 0:
                    time.sleep(0.5)
                    print ("Policy: %s, Experience count: %d/%d, Round: %d/%d") % (policy_name[policy[policy_num] - 1], experience_num + 1, total_experience_num, attempt, total_attempt_in_each_experience)
                    print ("Reward mean of each arm: ")
                    print (each_arm_mean)
                    print ("Empirical mean : ")
                    print (each_arm_empirical_reward_sum/each_arm_count)
                    print ("The number of times each arm is selected: ")
                    print (each_arm_count)
                    print ("\n")

            # To calculate EXP3 regret
            for i in xrange(n_arm):
                to_find_max_sum_reward[i] += each_arm_reward[i]

            if attempt % 1 == 0:

                if policy[policy_num] == 1 or policy[policy_num] == 2:
                    tmp_regret = cal_regret(attempt, each_arm_count, each_arm_mean)
                    upperbound = cal_regret_upperbound(policy[policy_num], attempt, n_arm, each_arm_mean)
                    regret_n_upperbound[1, attempt] = upperbound

                tmp_regret_save[experience_num,attempt] = tmp_regret

            # cumulative attempt count save
            if attempt%n_arm==0:
                for i in xrange(n_arm):
                    y[i, attempt/n_arm] = each_arm_count[i]

        print ("regret")
        print tmp_regret

        tmp_y_save[experience_num,:,:] = y

    # Print the cumulative number of times each arm is played ###############
    cumulative_count = np.mean(tmp_y_save, axis=0)
    x = np.arange(start=0,stop=total_attempt_in_each_experience+n_arm,step=n_arm)
    for i in xrange(n_arm):
        label = 'arm'+str(i+1)
        plt.plot(x,cumulative_count[i,:], label=label)
    plt.legend(loc='upper left')
    plt.title('Cumulative number of times each arm is played')
    plt.xlabel('Round n')
    plt.grid(True)
    plt.show()
    # #####################################################################

# Compare regrets of KL-UCB and UCB ###############################
#     regret_n_upperbound[0, :] = np.mean(tmp_regret_save, axis=0)
#     compare_regret[policy[policy_num]-1,:] = regret_n_upperbound[0,:]
#
# x = np.arange(start=0, stop=total_attempt_in_each_experience + 1, step=1)
# plt.plot(x, compare_regret[0, :], label='Regret of KL-UCB')
# plt.plot(x, compare_regret[1, :], label='Regret of UCB')
# plt.grid(True)
# plt.title('Regrets of KL-UCB and UCB')
# plt.xlabel('Round n')
# plt.legend(loc='upper left')
# plt.show()
#####################################################################