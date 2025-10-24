#!/usr/bin/env python3

import os
import argparse
import json
import math
import matplotlib.pyplot as plt

###############################################################################
# The next functions are already implemented for your convenience
#
# In all the functions in this stub file, `game` is the parsed input game json
# file, whereas `tfsdp` is either `game["decision_problem_pl1"]` or
# `game["decision_problem_pl2"]`.
#
# See the homework handout for a description of each field.


def get_sequence_set(tfsdp):
    """Returns a set of all sequences in the given tree-form sequential decision
    process (TFSDP)"""

    sequences = set()
    for node in tfsdp:
        if node["type"] == "decision":
            for action in node["actions"]:
                sequences.add((node["id"], action))
    return sequences


def is_valid_RSigma_vector(tfsdp, obj):
    """Checks that the given object is a dictionary keyed on the set of sequences
    of the given tree-form sequential decision process (TFSDP)"""

    sequence_set = get_sequence_set(tfsdp)
    return isinstance(obj, dict) and obj.keys() == sequence_set


def assert_is_valid_sf_strategy(tfsdp, obj):
    """Checks whether the given object `obj` represents a valid sequence-form
    strategy vector for the given tree-form sequential decision process
    (TFSDP)"""

    if not is_valid_RSigma_vector(tfsdp, obj):
        print("The sequence-form strategy should be a dictionary with key set equal to the set of sequences in the game")
        os.exit(1)
    for node in tfsdp:
        if node["type"] == "decision":
            parent_reach = 1.0
            if node["parent_sequence"] is not None:
                parent_reach = obj[node["parent_sequence"]]
            if abs(sum([obj[(node["id"], action)] for action in node["actions"]]) - parent_reach) > 1e-3:
                print(
                    "At node ID %s the sum of the child sequences is not equal to the parent sequence", node["id"])


def best_response_value(tfsdp, utility):
    """Computes the value of max_{x in Q} x^T utility, where Q is the
    sequence-form polytope for the given tree-form sequential decision
    process (TFSDP)"""

    assert is_valid_RSigma_vector(tfsdp, utility)

    utility_ = utility.copy()
    utility_[None] = 0.0
    for node in tfsdp[::-1]:
        if node["type"] == "decision":
            max_ev = max([utility_[(node["id"], action)]
                         for action in node["actions"]])
            utility_[node["parent_sequence"]] += max_ev
    return utility_[None]


def compute_utility_vector_pl1(game, sf_strategy_pl2):
    """Returns A * y, where A is the payoff matrix of the game and y is
    the given strategy for Player 2"""

    assert_is_valid_sf_strategy(
        game["decision_problem_pl2"], sf_strategy_pl2)

    sequence_set = get_sequence_set(game["decision_problem_pl1"])
    utility = {sequence: 0.0 for sequence in sequence_set}
    for entry in game["utility_pl1"]:
        utility[entry["sequence_pl1"]] += entry["value"] * \
            sf_strategy_pl2[entry["sequence_pl2"]]

    assert is_valid_RSigma_vector(game["decision_problem_pl1"], utility)
    return utility


def compute_utility_vector_pl2(game, sf_strategy_pl1):
    """Returns -A^transpose * x, where A is the payoff matrix of the
    game and x is the given strategy for Player 1"""

    assert_is_valid_sf_strategy(
        game["decision_problem_pl1"], sf_strategy_pl1)

    sequence_set = get_sequence_set(game["decision_problem_pl2"])
    utility = {sequence: 0.0 for sequence in sequence_set}
    for entry in game["utility_pl1"]:
        utility[entry["sequence_pl2"]] -= entry["value"] * \
            sf_strategy_pl1[entry["sequence_pl1"]]

    assert is_valid_RSigma_vector(game["decision_problem_pl2"], utility)
    return utility


def gap(game, sf_strategy_pl1, sf_strategy_pl2):
    """Computes the saddle point gap of the given sequence-form strategies
    for the players"""

    assert_is_valid_sf_strategy(
        game["decision_problem_pl1"], sf_strategy_pl1)
    assert_is_valid_sf_strategy(
        game["decision_problem_pl2"], sf_strategy_pl2)

    utility_pl1 = compute_utility_vector_pl1(game, sf_strategy_pl2)
    utility_pl2 = compute_utility_vector_pl2(game, sf_strategy_pl1)

    return (best_response_value(game["decision_problem_pl1"], utility_pl1)
            + best_response_value(game["decision_problem_pl2"], utility_pl2))


###########################################################################
# Starting from here, you should fill in the implementation of the
# different functions


def expected_utility_pl1(game, sf_strategy_pl1, sf_strategy_pl2):
    """Returns the expected utility for Player 1 in the game, when the two
    players play according to the given strategies"""

    assert_is_valid_sf_strategy(
        game["decision_problem_pl1"], sf_strategy_pl1)
    assert_is_valid_sf_strategy(
        game["decision_problem_pl2"], sf_strategy_pl2)

    # FINISH
    utility_pl1 = compute_utility_vector_pl1(game, sf_strategy_pl2)
    utility = sum(sf_strategy_pl1[seq] * utility_pl1[seq] for seq in utility_pl1)
    return utility


def uniform_sf_strategy(tfsdp):
    """Returns the uniform sequence-form strategy for the given tree-form
    sequential decision process"""

    # FINISH
    seqs = get_sequence_set(tfsdp)
    x = {seq: 0.0 for seq in seqs}
    
    for node in tfsdp:
        if node["type"] != "decision":
            continue
        
        if node["parent_sequence"] is None:
            parent_reach = 1.0
        else:
            parent_reach = x[node["parent_sequence"]]
            
        num_actions = len(node["actions"])
        if num_actions > 0:
            for action in node["actions"]:
                x[(node["id"], action)] = parent_reach / num_actions
    
    assert_is_valid_sf_strategy(tfsdp, x)
    return x


class RegretMatching(object):
    def __init__(self, action_set):
        self.action_set = set(action_set)

        # FINISH
        self.regret_sum = {action: 0.0 for action in action_set}
        self.last_strategy = None

    def next_strategy(self):
        
        # FINISH
        positive_regrets = {action: max(0.0, self.regret_sum[action]) for action in self.action_set}
        total_positive_regret = sum(positive_regrets.values())
        
        if total_positive_regret <= 1e-8:
            strategy = {action: 1.0 / len(self.action_set) for action in self.action_set}
        else:
            strategy = {action: positive_regrets[action] / total_positive_regret for action in self.action_set}
        
        self.last_strategy = strategy
        return strategy

    def observe_utility(self, utility):
        assert isinstance(utility, dict) and utility.keys() == self.action_set

        # FINISH
        if self.last_strategy is None:
            self.last_strategy = {action: 1.0 / len(self.action_set) for action in self.action_set}
        
        inner_product = sum(self.last_strategy[action] * utility[action] for action in self.action_set)
        for action in self.action_set:
            self.regret_sum[action] += utility[action] - inner_product


class RegretMatchingPlus(object):
    def __init__(self, action_set):
        self.action_set = set(action_set)

        # FINISH
        self.regret_sum = {action: 0.0 for action in action_set}
        self.last_strategy = None

    def next_strategy(self):
        
        # FINISH
        regrets = {action: self.regret_sum[action] for action in self.action_set}
        total_regret = sum(regrets.values())
        
        if total_regret <= 1e-8:
            strategy = {action: 1.0 / len(self.action_set) for action in self.action_set}
        else:
            strategy = {action: regrets[action] / total_regret for action in self.action_set}
            
        self.last_strategy = strategy
        return strategy

    def observe_utility(self, utility):
        assert isinstance(utility, dict) and utility.keys() == self.action_set

        # FINISH
        if self.last_strategy is None:
            self.last_strategy = {action: 1.0 / len(self.action_set) for action in self.action_set}
        
        inner_product = sum(self.last_strategy[action] * utility[action] for action in self.action_set)
        for action in self.action_set:
            self.regret_sum[action] = max(0.0, self.regret_sum[action] + utility[action] - inner_product)


class Cfr(object):
    def __init__(self, tfsdp, rm_class=RegretMatching):
        self.tfsdp = tfsdp
        self.local_regret_minimizers = {}
        self.last_strategies = {}
        self.children = {}

        # For each decision point, we instantiate a local regret minimizer
        for node in tfsdp:
            if node["type"] == "decision":
                self.local_regret_minimizers[node["id"]] = rm_class(
                    node["actions"])
                
        for node in tfsdp:
            parent_edge = node.get("parent_edge", None)
            if parent_edge is not None:
                self.children[parent_edge] = node["id"]

    def next_strategy(self):
        
        # FINISH
        seqs = get_sequence_set(self.tfsdp)
        x = {seq: 0.0 for seq in seqs}
        
        for node in self.tfsdp:
            if node["type"] != "decision":
                continue
            
            strategy = self.local_regret_minimizers[node["id"]].next_strategy()
            self.last_strategies[node["id"]] = strategy
            
            parent_reach = 1.0
            if node["parent_sequence"] is not None:
                parent_reach = x[node["parent_sequence"]]
            
            for action in node["actions"]:
                x[(node["id"], action)] = parent_reach * strategy[action]
                
        assert_is_valid_sf_strategy(self.tfsdp, x)
        return x

    def observe_utility(self, utility):
        
        # FINISH
        V = {None: 0.0}
        for node in self.tfsdp[::-1]:
            
            if node["type"] == "decision":
                j = node["id"]
                b = self.last_strategies.get(j, None)
                
                if b is None:
                    b = {action: 1.0 / len(node["actions"]) for action in node["actions"]}
                
                V_j = 0.0
                for action in node["actions"]:
                    child_id = self.children.get((j, action), None)
                    V_child = 0.0 if child_id is None else V.get(child_id, 0.0)
                    V_j += b[action] * (utility[(j, action)] + V_child)
                V[j] = V_j

            else:
                k = node["id"]
                V_k = 0.0
                for signal in node["signals"]:
                    child_id = self.children.get((k, signal), None)
                    V_child = 0.0 if child_id is None else V.get(child_id, 0.0)
                    V_k += V_child
                V[k] = V_k
        
        for node in self.tfsdp:
            if node["type"] != "decision":
                continue
            
            j = node["id"]
            u = {}
            for action in node["actions"]:
                child_id = self.children.get((j, action), None)
                V_child = 0.0 if child_id is None else V.get(child_id, 0.0)
                u[action] = utility[(j, action)] + V_child
            self.local_regret_minimizers[j].observe_utility(u)


def dict_add_in_place(dst, src, weight=1.0):
    for key, value in src.items():
        dst[key] = dst.get(key, 0.0) + weight * value
        

def dict_scale(dict, scale):
    return {key: value * scale for key, value in dict.items()}


def apply_dcfr_discount(cfr, t, alpha, beta):
    if alpha == 0.0 and beta == 0.0:
        return
    
    pos_scale = (t ** alpha) / (t ** alpha + 1.0)
    neg_scale = (t ** beta) / (t ** beta + 1.0)
    
    for rm in cfr.local_regret_minimizers.values():
        regret_sum = rm.regret_sum
        
        for (action, regret) in regret_sum.items():
            if regret >= 0.0:
                regret_sum[action] = regret * pos_scale
            else:
                regret_sum[action] = regret * neg_scale


def plotter_v(v):
    T = range(1, len(v) + 1)
    plt.figure(figsize=(7, 4))
    plt.plot(T, v, linewidth=2)
    plt.xlabel("Iteration T")
    plt.ylabel("Expected utility v_t")
    plt.title("CFR: Leduc Poker")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()
    
    
def plotter_gap(gap):
    T = range(1, len(gap) + 1)
    plt.figure(figsize=(7, 4))
    plt.plot(T, gap, linewidth=2, color='red')
    plt.xlabel("Iteration T")
    plt.ylabel("Saddle point gap g_t")
    plt.title("CFR: Leduc Poker")
    # plt.yscale("log")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()
    
    
def plotter_v_vs_baseline(v, baseline_v):
    T1 = range(1, len(v) + 1)
    T2 = range(1, len(baseline_v) + 1)
    plt.figure(figsize=(7, 4))
    plt.plot(T1, v, linewidth=2, label='Variant')
    plt.plot(T2, baseline_v, linestyle='--', color='r', linewidth=2, label='Baseline')
    plt.xlabel("Iteration T")
    plt.ylabel("Expected utility v_t")
    plt.title("CFR")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()
    
    
def plotter_gap_vs_baseline(gap, baseline_gap):
    T1 = range(1, len(gap) + 1)
    T2 = range(1, len(baseline_gap) + 1)
    plt.figure(figsize=(7, 4))
    plt.plot(T1, gap, linewidth=2, label='Variant')
    plt.plot(T2, baseline_gap, linestyle='--', color='r', linewidth=2, label='Baseline')
    plt.xlabel("Iteration T")
    plt.ylabel("Saddle point gap g_t")
    plt.title("CFR")
    # plt.yscale("log")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()
    

def plotter_v_aggregate(v_list, labels):
    T = range(1, len(v_list[0]) + 1)
    plt.figure(figsize=(7, 4))
    
    for v, label in zip(v_list, labels):
        plt.plot(T, v, linewidth=2, label=label)
        
    plt.xlabel("Iteration T")
    plt.ylabel("Expected utility v_t")
    plt.title("CFR")
    # plt.yscale("log")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()
    

def plotter_gap_aggregate(gap_list, labels):
    T = range(1, len(gap_list[0]) + 1)
    plt.figure(figsize=(7, 4))
    
    for gap, label in zip(gap_list, labels):
        plt.plot(T, gap, linewidth=2, label=label)
        
    plt.xlabel("Iteration T")
    plt.ylabel("Saddle point gap g_t")
    plt.title("CFR")
    # plt.yscale("log")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()
    

def plotter_gap_aggregate_log(gap_list, labels):
    T = range(1, len(gap_list[0]) + 1)
    plt.figure(figsize=(7, 4))
    
    for gap, label in zip(gap_list, labels):
        plt.plot(T, gap, linewidth=2, label=label)
        
    plt.xlabel("Iteration T")
    plt.ylabel("Saddle point gap g_t")
    plt.title("CFR")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


def solve_problem_3_1(game, T=1000):
    
    # FINISH
    v_list = []
    y_uniform = uniform_sf_strategy(game["decision_problem_pl2"])
    u = compute_utility_vector_pl1(game, y_uniform)
    cfr_pl1 = Cfr(game["decision_problem_pl1"], rm_class=RegretMatching)
    
    sequences = get_sequence_set(game["decision_problem_pl1"])
    x_sum = {seq: 0.0 for seq in sequences}
    
    for t in range(1, T + 1):
        x_t = cfr_pl1.next_strategy()
        cfr_pl1.observe_utility(u)
        
        dict_add_in_place(x_sum, x_t, weight=1.0)
        
        x_bar = dict_scale(x_sum, 1.0 / t)
        v_t = expected_utility_pl1(game, x_bar, y_uniform)
        v_list.append(v_t)
        
        if t % 100 == 0 or t == 1 or t == T:
            print("Iteration %d. Expected utility = %.6f" % (t, v_t))
        
    return v_list


def solve_problem_3_2(game, T=1000):
    
    # FINISH
    v_list = []
    gap_list = []
    
    cfr_pl1 = Cfr(game["decision_problem_pl1"], rm_class=RegretMatching)
    cfr_pl2 = Cfr(game["decision_problem_pl2"], rm_class=RegretMatching)
    
    sequences_pl1 = get_sequence_set(game["decision_problem_pl1"])
    sequences_pl2 = get_sequence_set(game["decision_problem_pl2"])
    x_sum = {seq: 0.0 for seq in sequences_pl1}
    y_sum = {seq: 0.0 for seq in sequences_pl2}
    
    for t in range(1, T + 1):
        x_t = cfr_pl1.next_strategy()
        y_t = cfr_pl2.next_strategy()
        
        u_pl1 = compute_utility_vector_pl1(game, y_t)
        u_pl2 = compute_utility_vector_pl2(game, x_t)
        
        cfr_pl1.observe_utility(u_pl1)
        cfr_pl2.observe_utility(u_pl2)
        
        dict_add_in_place(x_sum, x_t, weight=1.0)
        dict_add_in_place(y_sum, y_t, weight=1.0)
        
        x_bar = dict_scale(x_sum, 1.0 / t)
        y_bar = dict_scale(y_sum, 1.0 / t)
        
        v_t = expected_utility_pl1(game, x_bar, y_bar)
        g_t = gap(game, x_bar, y_bar)
        
        v_list.append(v_t)
        gap_list.append(g_t)
        
        if t % 100 == 0 or t == 1 or t == T:
            print("Iteration %d. Expected utility = %.6f. Gap = %.6f" % (t, v_t, g_t))

    return v_list, gap_list


def solve_problem_3_3(game, T=1000, alpha=0.0, beta=0.0, gamma=0.0, use_rm_plus=True):
    
    # FINISH
    print("Parameters: alpha =", alpha, "beta =", beta, "gamma =", gamma, "use_rm_plus =", use_rm_plus)
    v_list = []
    gap_list = []
    
    rm_class = RegretMatchingPlus if use_rm_plus else RegretMatching
    cfr_pl1 = Cfr(game["decision_problem_pl1"], rm_class=rm_class)
    cfr_pl2 = Cfr(game["decision_problem_pl2"], rm_class=rm_class)
    
    keep_last = (gamma == float('inf'))
    if keep_last:
        print("Using last iterate only (gamma = inf)")
    
    else:
        sequences_pl1 = get_sequence_set(game["decision_problem_pl1"])
        sequences_pl2 = get_sequence_set(game["decision_problem_pl2"])
        x_wsum = {seq: 0.0 for seq in sequences_pl1}
        y_wsum = {seq: 0.0 for seq in sequences_pl2}
        wsum = 0.0
    
    for t in range(1, T + 1):
        
        y_t = cfr_pl2.next_strategy()
        u_pl1 = compute_utility_vector_pl1(game, y_t)
        apply_dcfr_discount(cfr_pl1, t, alpha, beta)
        cfr_pl1.observe_utility(u_pl1)
        
        x_tp1 = cfr_pl1.next_strategy()
        u_pl2 = compute_utility_vector_pl2(game, x_tp1)
        apply_dcfr_discount(cfr_pl2, t, alpha, beta)
        cfr_pl2.observe_utility(u_pl2)
        
        if keep_last:
            x_bar = x_tp1
            y_bar = y_t
        
        else:
            w_t = t ** gamma
            dict_add_in_place(x_wsum, x_tp1, weight=w_t)
            dict_add_in_place(y_wsum, y_t, weight=w_t)
    
            wsum += w_t
            x_bar = dict_scale(x_wsum, 1.0 / wsum)
            y_bar = dict_scale(y_wsum, 1.0 / wsum)
        
        v_t = expected_utility_pl1(game, x_bar, y_bar)
        g_t = gap(game, x_bar, y_bar)
        
        v_list.append(v_t)
        gap_list.append(g_t)
        
        if t % 100 == 0 or t == 1 or t == T:
            print("Iteration %d. Expected utility = %.6f. Gap = %.6f" % (t, v_t, g_t))
    
    return v_list, gap_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Problem 3 (CFR)')
    parser.add_argument("--game", help="Path to game file")
    parser.add_argument("--problem", choices=["3.1", "3.2", "3.3"])
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--use_rm_plus", action="store_true", default=False)

    args = parser.parse_args()
    print("Reading game path %s..." % args.game)

    game = json.load(open(args.game))

    # Convert all sequences from lists to tuples
    for tfsdp in [game["decision_problem_pl1"], game["decision_problem_pl2"]]:
        for node in tfsdp:
            if isinstance(node["parent_edge"], list):
                node["parent_edge"] = tuple(node["parent_edge"])
            if "parent_sequence" in node and isinstance(node["parent_sequence"], list):
                node["parent_sequence"] = tuple(node["parent_sequence"])
    for entry in game["utility_pl1"]:
        assert isinstance(entry["sequence_pl1"], list)
        assert isinstance(entry["sequence_pl2"], list)
        entry["sequence_pl1"] = tuple(entry["sequence_pl1"])
        entry["sequence_pl2"] = tuple(entry["sequence_pl2"])

    print("... done. Running code for Problem", args.problem)
    print("Alpha =", args.alpha, "Beta =", args.beta, "Gamma =", args.gamma, "T =", args.T, "Use RM Plus =", args.use_rm_plus)

    if args.problem == "3.1":
        solve_problem_3_1(game, T=args.T)
        v_list = solve_problem_3_1(game, T=args.T)
        plotter_v(v_list)
        
    elif args.problem == "3.2":
        v_list, gap_list = solve_problem_3_2(game, T=args.T)
        plotter_v(v_list)
        plotter_gap(gap_list)
    
    else:
        assert args.problem == "3.3"
        v_cfr, gap_cfr = solve_problem_3_2(game, T=args.T) # CFR
        v_cfrp, gap_cfrp = solve_problem_3_3(game, T=args.T, alpha=0.0, beta=0.0, gamma=1.0, use_rm_plus=True) # CFR+
        v_dcfr, gap_dcfr = solve_problem_3_3(game, T=args.T, alpha=1.5, beta=0.0, gamma=2.0, use_rm_plus=False) # DCFR
        v_pcfrp_2, gap_pcfrp_2 = solve_problem_3_3(game, T=args.T, alpha=0.0, beta=0.0, 
                                                   gamma=2.0, use_rm_plus=True) # PCFR+ gamma=2
        v_pcfrp_inf, gap_pcfrp_inf = solve_problem_3_3(game, T=args.T, alpha=0.0, beta=0.0, 
                                                       gamma=float('inf'), use_rm_plus=True) # PCFR+ last iterate
        
        plotter_v_aggregate([v_cfr, v_cfrp, v_dcfr, v_pcfrp_2, v_pcfrp_inf],
                             ['CFR', 'CFR+', 'DCFR', 'PCFR+ (gamma=2)', 'PCFR+ (last)'])
        plotter_gap_aggregate([gap_cfr, gap_cfrp, gap_dcfr, gap_pcfrp_2, gap_pcfrp_inf],
                             ['CFR', 'CFR+', 'DCFR', 'PCFR+ (gamma=2)', 'PCFR+ (last)'])
        plotter_gap_aggregate_log([gap_cfr, gap_cfrp, gap_dcfr, gap_pcfrp_2, gap_pcfrp_inf],
                             ['CFR', 'CFR+', 'DCFR', 'PCFR+ (gamma=2)', 'PCFR+ (last)'])
        
