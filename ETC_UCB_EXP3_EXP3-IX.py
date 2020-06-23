import numpy as np
import math
import random
import matplotlib.pyplot as plt
import time

def generate_arm_reward(n_arm, each_arm_mean, each_arm_variance):
    each_arm_reward = []
    for i in xrange(n_arm):
        each_arm_reward.append(np.random.normal(loc = each_arm_mean[i], scale = math.sqrt(each_arm_variance)))
    return each_arm_reward

# policy
def take_action(attempt, policy, n_arm, each_arm_empirical_reward_sum, each_arm_count, previous_action_num, weight, each_arm_reward):
    if policy == 1: # ETC
        if attempt <= 200*n_arm: # Round robin
            action_num = attempt % n_arm
            previous_action_num = action_num
        if attempt == 200*n_arm+1:
            empirical_mean = each_arm_empirical_reward_sum / each_arm_count
            action_num = np.argmax(empirical_mean)
            previous_action_num = action_num

        else:
            action_num = previous_action_num
            previous_action_num = action_num

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

    if policy == 3: # EXP3
        eta = math.sqrt(math.log(n_arm)/(n_arm * attempt))

        # Calculate sampling distribution
        sampling_dist = []
        sampling_dist_denominator = 0
        for i in xrange(n_arm):
            try:
                ans = math.exp(eta*weight[i])
            except OverflowError:
                ans = float('inf')
            sampling_dist_denominator += ans
        for i in xrange(n_arm):
            try:
                ans = math.exp(eta*weight[i])
            except OverflowError:
                ans = float('inf')
            sampling_dist.append(ans/sampling_dist_denominator)

        # Sample an action
        random_value = random.random()
        i = 0
        cumul_sampling_dist = 0
        while 1:
            cumul_sampling_dist += sampling_dist[i]
            if random_value <= cumul_sampling_dist:
                action_num = i
                break
            else:
                i += 1

        # Update weight
        for i in xrange(n_arm):
            if i == action_num:
                weight[i] += 1
                weight[i] -= (1-each_arm_reward[i])/sampling_dist[i]
            else:
                weight[i] += 1

    if policy == 4: # EXP3-IX
        eta_1 = math.sqrt(2*math.log(n_arm+1) / (n_arm * attempt))
        gamma = eta_1/2.0

        # Calculate sampling distribution
        sampling_dist = []
        sampling_dist_denominator = 0
        for i in xrange(n_arm):
            sampling_dist_denominator += math.exp((-eta_1) * weight[i])
        for i in xrange(n_arm):
            sampling_dist.append(math.exp((-eta_1) * weight[i]) / sampling_dist_denominator)

        # Sample an action
        random_value = random.random()
        i = 0
        cumul_sampling_dist = 0
        while 1:
            cumul_sampling_dist += sampling_dist[i]
            if random_value <= cumul_sampling_dist:
                action_num = i
                break
            else:
                i += 1

        # Update weight
        for i in xrange(n_arm):
            if i == action_num:
                weight[i] += (1 - each_arm_reward[i]) / (sampling_dist[i] + gamma)

    return action_num, previous_action_num, weight

def cal_regret(attempt, each_arm_count, each_arm_mean):
    best_arm = np.argmax(each_arm_mean)
    best = (attempt) * each_arm_mean[best_arm]

    empirical_reward_sum = 0
    for i in xrange(len(each_arm_count)):
        empirical_reward_sum += each_arm_count[i]*each_arm_mean[i]

    regret = best - empirical_reward_sum

    return regret

def cal_regret_EXP(to_find_max_sum_reward, each_arm_empirical_reward_sum):
    chosen_action_reward_sum = sum(each_arm_empirical_reward_sum)
    regret = max(to_find_max_sum_reward) - chosen_action_reward_sum

    return regret

def cal_regret_upperbound(policy, n, k, each_arm_mean):
    if policy == 1:
        # Theorem 6.1
        exploration_num = 200
        first_term_1 = min(exploration_num, math.ceil(n/float(k)))
        best = max(each_arm_mean)
        gap = []
        for i in xrange(k):
            gap.append(best-each_arm_mean[i])
        first_term_2 = sum(gap)
        second_term_1 = max(0,n-exploration_num*k)
        second_term_2 = 0
        for i in xrange(k):
            second_term_2+=gap[i]*math.exp(-(exploration_num*gap[i]*gap[i]/4))
        upperbound = first_term_1*first_term_2 + second_term_1*second_term_2

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

total_attempt_in_each_experience = 10000
total_experience_num = 10

# arm info
n_arm = 10
each_arm_mean = np.random.uniform(low=0, high=1, size=(n_arm,))
each_arm_variance = 1

# 1:ETC, 2:UCB, 3:EXP3, 4:EXP3-IX
policy = [1, 2, 3, 4]
policy_name = ['ETC', 'UCB', 'EXP3', 'EXP3-IX']

compare_regret = np.zeros(shape=(2, total_attempt_in_each_experience + 1))

for policy_num in xrange(len(policy)):

    regret_n_upperbound = np.zeros(shape=(2, total_attempt_in_each_experience + 1))
    tmp_regret_save = np.zeros(shape=(total_experience_num, total_attempt_in_each_experience + 1))
    tmp_y_save = np.zeros(shape=(total_experience_num, n_arm, total_attempt_in_each_experience/n_arm+1))

    for experience_num in xrange(total_experience_num):
        each_arm_count = np.zeros(shape=(n_arm,), dtype=np.float64)
        each_arm_empirical_reward_sum = np.zeros(shape=(n_arm,), dtype=np.float64)
        to_find_max_sum_reward = np.zeros(shape=(n_arm,), dtype=np.float64)

        weight = np.zeros(shape=(n_arm,), dtype=np.float64)

        previous_action_num = 0

        y = np.zeros(shape=(n_arm,total_attempt_in_each_experience/n_arm+1))

        for attempt in xrange(1, total_attempt_in_each_experience+1):
            each_arm_reward = generate_arm_reward(n_arm, each_arm_mean, each_arm_variance)

            action_num, previous_action_num, weight = take_action(attempt, policy[policy_num], n_arm, each_arm_empirical_reward_sum, each_arm_count, previous_action_num, weight, each_arm_reward)

            each_arm_count[action_num] += 1
            each_arm_empirical_reward_sum[action_num] += each_arm_reward[action_num]


            if (experience_num + 1) % 10 == 0:
                if attempt % 1000 == 0:
                    time.sleep(0.5)
                    print ("Policy: [%s]") % policy_name[policy[policy_num] - 1]
                    print ("Experience count: %d/%d, Round: %d/%d") % (experience_num + 1, total_experience_num, attempt, total_attempt_in_each_experience)
                    print ("Reward mean of each arm: ")
                    print (each_arm_mean)
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
                if policy[policy_num] == 3 or policy[policy_num] == 4:
                    tmp_regret = cal_regret_EXP(to_find_max_sum_reward, each_arm_empirical_reward_sum)

                tmp_regret_save[experience_num,attempt] = tmp_regret

            # cumulative attempt count save
            if attempt%n_arm==0:
                for i in xrange(n_arm):
                    y[i, attempt/n_arm] = each_arm_count[i]

        tmp_y_save[experience_num,:,:] = y

    # Print the cumulative number of times each arm is played ###############
    # cumulative_count = np.mean(tmp_y_save, axis=0)
    # x = np.arange(start=0,stop=total_attempt_in_each_experience+n_arm,step=n_arm)
    # for i in xrange(n_arm):
    #     label = 'arm'+str(i+1)
    #     plt.plot(x,cumulative_count[i,:], label=label)
    # plt.legend(loc='upper left')
    # plt.title('Cumulative number of times each arm is played')
    # plt.xlabel('Round n')
    # plt.grid(True)
    # plt.ylim(0,550)
    # plt.show()
    # #####################################################################

    # Print regret and upper bound ######################################
    # regret_n_upperbound[0, :] = np.mean(tmp_regret_save, axis=0)
    # x = np.arange(start=0, stop=total_attempt_in_each_experience+1, step=1)
    # plt.plot(x, regret_n_upperbound[0, :], label='Regret')
    # plt.plot(x, regret_n_upperbound[1, :], label='Upper bound by Theorem 6.1', ls='--')
    # plt.grid(True)
    # plt.title('Regret analysis')
    # plt.xlabel('Round n')
    # plt.legend(loc='upper left')
    # plt.show()
    #####################################################################

    # Compare regrets of EXP3 and EXP3-IX ###############################
    # regret_n_upperbound[0, :] = np.mean(tmp_regret_save, axis=0)
    # if policy[policy_num] == 3 or policy[policy_num] == 4:
    #     compare_regret[policy[policy_num]-3,:] = regret_n_upperbound[0,:]
    #
    # x = np.arange(start=0, stop=total_attempt_in_each_experience + 1, step=1)
    # plt.plot(x, compare_regret[0, :], label='Regret of EXP3')
    # plt.plot(x, compare_regret[1, :], label='Regret of EXP3-IX')
    # plt.grid(True)
    # plt.title('Compare regrets of EXP3 and EXP3-IX')
    # plt.xlabel('Round n')
    # plt.legend(loc='upper left')
    # plt.show()
    #####################################################################