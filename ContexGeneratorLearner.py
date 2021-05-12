import numpy as np

class ContextGeneratorLEarner(object):
    def __init__(self, arms, customer_classes, bandit_class, hoeffding_confidence):
        self.arms = arms
        self.bandit_class = bandit_class
        self.customer_classes = customer_classes
        self.features = [customer_class.feature_values for customer_class in self.customer_classes] 
        self.hoeffding_confidence = hoeffding_confidence
        self.class_to_context = [0 for _ in self.customer_classes] 
        self.context_to_class = [customer_class.feature_values for customer_class in self.customer_classes]
        self.context_to_bandit = [self.bandit_class(self.arms)]
        self.all_rewards_per_day = []
        print(self.features)

    def pull_arm(self):
        arms = [None for _ in self.customer_classes]
        for context_idx, bandit in enumerate(self.context_to_bandit):
            price =bandit.pull_arm()
            for class_idx in np.argwhere(np.array(self.class_to_context) == context_idx).reshape(-1):
                arms[class_idx] = price
        return arms

    def update(self, rewards):
        self.all_rewards_per_day.append(rewards)
        rewards_per_context = [[] for _ in range(len(self.context_to_class))]
        for reward in rewards:
            class_idx = self.customer_classes.index(reward.customer_class)
            context_idx = self.class_to_context[class_idx]
            rewards_per_context[context_idx].append(reward)
        for context_idx, bandit in enumerate(self.context_to_bandit):
            bandit.update(rewards_per_context[context_idx])

    def update_contexts(self):
        bandit = self.bandit_class(self.arms)
        for daily_reward in self.all_rewards_per_day:
            bandit.update(daily_reward)
        self.context_to_bandit = self.generate_contex_tree(bandit, self.features)

    def generate_contex_tree(self, bandit, features_set):
        split_values = []
        context_bandits = []
        print('feature_set', features_set)
        print('bandit', bandit)
        for feature_idx in range(2):
            if len(set(map(lambda x: x[feature_idx], features_set))) > 1:
                split_object = self.split_value_by_features(bandit, features_set, feature_idx)
                split_values.append(split_object[0])
                context_bandits.append(split_object[1:])
            else:
                split_values.append(0)
                context_bandits.append([bandit])
        max_value = max(split_values)
        max_idx = split_values.index(max_value)
        print('values', split_values)
        if max_value > 0:
            left_context = list(filter(lambda x: x[max_idx] == 0, features_set))
            right_context = list(filter(lambda x: x[max_idx] == 1, features_set))
            print('left-right', left_context, right_context)
            left_node = [context_bandits[max_idx][0]]
            right_node = [context_bandits[max_idx][1]]
            if len(left_context) > 1:
                left_node = self.generate_contex_tree(left_node[0], left_context)
            if len(right_context) > 1:
                right_node = self.generate_contex_tree(right_node[0], right_context)
            return [*left_node, *right_node]
        return context_bandits[0]


    def split_value_by_features(self, total_bandit, features_set, feature_idx):
        left_context = list(filter(lambda x: x[feature_idx] == 0, features_set))
        right_context = list(filter(lambda x: x[feature_idx] == 1, features_set))

        left_context_rewards = self.get_reward_by_features(left_context)

        right_context_rewards = self.get_reward_by_features(right_context)

        flatten_left_rewards = [rewards for daily_rewards in left_context_rewards for rewards in daily_rewards]
        flatten_right_rewards = [rewards for daily_rewards in right_context_rewards for rewards in daily_rewards]

        p_left = self.get_hoeffding_lower_bound(len(flatten_left_rewards)/(len(flatten_left_rewards) + len(flatten_right_rewards)),
                                           len(flatten_left_rewards) + len(flatten_right_rewards),
                                           self.hoeffding_confidence)
        p_right = self.get_hoeffding_lower_bound(len(flatten_right_rewards)/(len(flatten_left_rewards) + len(flatten_right_rewards)),
                                           len(flatten_left_rewards) + len(flatten_right_rewards),
                                           self.hoeffding_confidence)
        
        optimal_arm = total_bandit.get_optimal_arm()
        total_expected_rewards = list(map(lambda x: x.conversion*optimal_arm, total_bandit.customers_per_arm[total_bandit.arms.index(optimal_arm)]))

        left_expected_rewards, left_bandit = self.get_expected_rewards_by_bandit(left_context_rewards)
        right_expected_rewards, right_bandit = self.get_expected_rewards_by_bandit(right_context_rewards)

        mu_total = self.get_hoeffding_lower_bound(np.mean(total_expected_rewards),
                                                 len(total_expected_rewards),
                                                 self.hoeffding_confidence)
        mu_left = self.get_hoeffding_lower_bound(np.mean(left_expected_rewards),
                                                 len(left_expected_rewards),
                                                 self.hoeffding_confidence)
        mu_right = self.get_hoeffding_lower_bound(np.mean(right_expected_rewards),
                                                 len(right_expected_rewards),
                                                 self.hoeffding_confidence)

        split_value = p_left*mu_left + p_right*mu_right

        return [split_value, left_bandit, right_bandit] if split_value > mu_total else [0, total_bandit]

        
    def get_reward_by_features(self, features_set):
        filtered_rewards = []
        for daily_rewards in self.all_rewards_per_day:
            filtered_rewards.append(list(filter(lambda x: x.customer_class.feature_values in features_set, daily_rewards)))
        return filtered_rewards


    def get_expected_rewards_by_bandit(self, rewards):
        bandit = self.bandit_class(self.arms)
        for daily_reward in rewards:
            bandit.update( daily_reward)
        optimal_arm = bandit.get_optimal_arm()
        best_rewards = list(map(lambda x: x.conversion*optimal_arm, bandit.customers_per_arm[bandit.arms.index(optimal_arm)]))
        return best_rewards, bandit

    def get_hoeffding_lower_bound(self, mean, cardinality, confidence):
        if cardinality == 0:
            return -np.inf
        return mean - np.sqrt(-np.log(confidence) / (2 * cardinality))