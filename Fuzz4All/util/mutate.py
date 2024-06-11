import numpy as np


class GA(object):
    def __init__(
        self,
        num_generated,
        mutator_selection_algo="heuristic",
        use_single_mutator=False,
        replace_type=None,
        mutator_set="all",
    ):
        """
            mutator_selection_algo:
                heuristic: heuristic functions
                random: uniform random
                epsgreedy: Epsilon Greedy (eps=0.1)
                ucb: Upper Confidence Bound
        """
        self.num_generated = num_generated
        self.mutator_selection_algo = mutator_selection_algo
        self.use_single_mutator = use_single_mutator
        self.replace_type = replace_type
        self.mutator_set = mutator_set

        self.num_valid = 0

        self._init_mutator()


    def _init_multi_arm(self):
        self.replace_type_p = {}
        for t in self.replace_type:
            # Initialize with 1 to avoid starving
            # Change initial state to 0.5 following the Thompson Sampling algorithm
            self.replace_type_p[t] = [1, 2]  # success / total
        self.epsilon = 0.1

    def _init_mutator(self):
        if self.use_single_mutator:
            self.replace_type = [self.replace_type]
        else:
            """
            用来做三种变异策略的对比用的(评估对比)
            """
            if self.mutator_set == "nomutate-existing":
                self.replace_type = [
                    "generete-new",
                    "semantic-equiv",
                ]
            elif self.mutator_set == "nogenerate-new":
                self.replace_type = [
                    "mutate-existing",
                    "semantic-equiv",
                ]
            elif self.mutator_set == "nosemantic-equiv":
                self.replace_type = [
                    "generete-new",
                    "mutate-existing",
                ]
            elif self.mutator_set == "all":
                self.replace_type = [
                    "generete-new",
                    "mutate-existing",
                    "semantic-equiv",
                ]
            else:
                print("Replace_type {self.replace_type} not supported.")
                exit(-1)

        if self.mutator_selection_algo == "heuristic":
            self.replace_type_p = {
                "generete-new": self.num_generated * 3,
                "mutate-existing": self.num_generated * 3,
                "semantic-equiv": self.num_generated * 3,
            }
        elif self.mutator_selection_algo in ["epsgreedy", "ucb", "ts"]:
            # Multi-Arm Bandit strategies
            self._init_multi_arm()


    def _compute_score(self, code):
        raise NotImplementedError

    def _select_mutator(self):
        if self.mutator_selection_algo == "heuristic":
            replace_type = np.random.choice(
                self.replace_type,
                1,
                p=[
                    self.replace_type_p[x] / sum(list(self.replace_type_p.values()))
                    for x in self.replace_type
                ],
            )[0]
            return replace_type
        elif self.mutator_selection_algo == "random":
            replace_type = np.random.choice(self.replace_type)
            return replace_type
        elif self.mutator_selection_algo == "epsgreedy":
            expl = np.random.uniform(0.0, 1.0)
            if expl > self.epsilon:  # exploit
                max_value = max(
                    [x[0] / x[1] for v, x in self.replace_type_p.items() if x[1] != 0],
                    default=0,
                )
                if max_value == 0:
                    replace_type = np.random.choice(self.replace_type)
                else:
                    replace_type = [
                        k
                        for k, v in self.replace_type_p.items()
                        if v[1] != 0 and v[0] / v[1] == max_value
                    ][0]
            else:  # explore
                replace_type = np.random.choice(self.replace_type)
            return replace_type
        elif self.mutator_selection_algo == "ucb":
            total_num = sum([x[1] for x in self.replace_type_p.values()])
            log_t_2 = 2.0 * np.log(total_num)
            # UCB1 score: mu(a) + sqrt(2 * log(t) / n_t(a))
            type_scores = [
                (v, x[0] / x[1] + np.sqrt(log_t_2 / x[1]))
                for v, x in self.replace_type_p.items()
            ]
            types, scores = list(zip(*type_scores))
            max_index = np.argmax(scores)
            max_score = scores[max_index]
            max_types = [t for t, score in type_scores if score >= max_score]
            return np.random.choice(max_types)
        elif self.mutator_selection_algo == "ts":
            scores = []
            for a in self.replace_type:
                alpha, n_t = self.replace_type_p[a]
                beta = n_t - alpha
                score_a = np.random.beta(alpha, beta)
                scores.append(score_a)
            max_index = np.argmax(scores)
            return self.replace_type[max_index]


    def _update_mutator(self, generations, replace_type):
        if self.mutator_selection_algo == "heuristic":
            # update the global counter
            # roughly that score increases when at least 1/4 of generation is valid and unique
            self.replace_type_p[replace_type] += len(generations) - 1 / 3 * (
                self.num_generated - len(generations)
            )
            self.replace_type_p[replace_type] = max(
                1, self.replace_type_p[replace_type]
            )
        elif self.mutator_selection_algo in ["epsgreedy", "ucb", "ts"]:
            # update the global counter
            self.replace_type_p[replace_type][0] += len(generations)
            self.replace_type_p[replace_type][1] += self.num_generated
        elif self.mutator_selection_algo == "random":
            pass
        else:
            raise NotImplementedError(
                "Operator selction algorithm {} not supported".format(
                    self.mutator_selection_algo
                )
            )

    def update(self, generations, replace_type):
        self._update_mutator(generations, replace_type)

    def get_p(self):
        if self.mutator_selection_algo == "heutistic":
            return [
                self.replace_type_p[x] / sum(list(self.replace_type_p.values()))
                for x in self.replace_type
            ]
        elif self.mutator_selection_algo == "random":
            return [1.0 / len(self.replace_type)] * len(self.replace_type)
        else:
            return self.replace_type_p

