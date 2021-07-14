import numpy as np

from main.contexts.PriceContextGenerator import PriceContextGenerator


class PriceBruteForceContextGenerator(PriceContextGenerator):
    """
    A context generator that uses brute force to find best context structure
    """
    def __init__(self, features, customer_classes, learner_class, arms, returns_horizon, confidence):
        """Class constructor

        Args:
            features (list): list of features for splitting contexts
            customer_classes (list): list of customer classes
            learner_class (PriceLearner): type of price learner used
            arms (list): set of arms to pull
            returns_horizon (integer): days horizon during which a customer can return to purchase after the first time
            confidence (float): Hoeffding confidence
        """

        super().__init__(features, customer_classes, learner_class, arms, returns_horizon, confidence)

    def get_best_contexts(self, incremental=False):
        """Get the best context structure

        Args:
            incremental (bool, optional): if False generate the context structure from scratch else start from the context structure active at the moment. Defaults to False.

        Returns:
            tuple: (list of context generated, list of learners for each context)
        """
        if len(self.customers_per_day) == 0:
            base_learner = self.generate_learner(self.customer_classes)
            self.contexts.append(self.customer_classes)
            self.learners.append(base_learner)
        else:
            base_contexts = self.contexts.copy() if incremental else [self.customer_classes]
            self.contexts = []
            self.learners = []
            for idx in range(len(base_contexts)):
                self.generate_contexts(base_contexts[idx])
        return self.contexts, self.learners

    def generate_contexts(self, base_contex):
        """Generate the best context structure from all possible ones

        Args:
            base_contex (list): context from which start
        """
        possible_contexts_split = []
        learners_per_split = []
        rewards_per_split = []
        for contexts_split in self.generate_partition(base_contex):
            if len(contexts_split) > 0:
                possible_contexts_split.append(contexts_split)
                learners = [self.generate_learner(context) for context in contexts_split]
                learners_per_split.append(learners)
                rewards_per_split.append(np.sum([self.get_context_reward_lower_bound(learner) for learner in learners]))
        best_split_idx = np.argmax(rewards_per_split)
        self.contexts.extend(possible_contexts_split[best_split_idx])
        self.learners.extend(learners_per_split[best_split_idx])

    def get_context_reward_lower_bound(self, learner):
        """Get the lower bound of the reward of a given learner

        Args:
            learner (PriceLearner): learner from which get the rewards

        Returns:
            np.Floating: lower bound of the reward
        """
        context_customers = np.sum([arm for arm in learner.samples_per_arm])
        total_customers = np.sum([len(daily_customers) for daily_customers in self.customers_per_day])
        
        best_arm = learner.get_optimal_arm()
        best_arm_idx = learner.arms.index(best_arm)
        best_arm_rewards = learner.rewards_per_arm[best_arm_idx]
        
        mean = np.mean(best_arm_rewards)
        std = np.std(best_arm_rewards)
        cardinality = len(best_arm_rewards)

        mu_val = self.get_mean_lower_bound(mean, std, cardinality)
        p_val = self.get_hoeffding_lower_bound(context_customers / total_customers, total_customers)
        return p_val*mu_val

    def generate_partition(self, context):
        """Generate all the possible context structures from the given one

        Args:
            context (list): list of customer classes from which generate all the the possible context structures

        Yields:
            list: list of contexts generated
        """

        if len(context) == 1:
            yield [context]
            return

        first = context[0]
        for smaller in self.generate_partition(context[1:]):
            for n, subset in enumerate(smaller):
                yield smaller[:n] + [[first] + subset] + smaller[n + 1:]
            yield [[first]] + smaller
