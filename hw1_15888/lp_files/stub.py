#!/usr/bin/env python3

import os
import argparse
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import gurobipy as gurobi
from gurobipy import GRB


def build_strategy_polytope(tfsdp):
    
    # Build decision points
    J = [] # Decision nodes
    A_j = {} # Action set at node j
    p_j = {} # Parent sequence at node j
    
    for node in tfsdp:
        if node["type"] == "decision":
            j = node["id"]
            J.append(j)
            A_j[j] = list(node["actions"])
            p_j[j] = node.get("parent_sequence", None)
    
    # Build all sequences of (j, a) avaliable to player        
    sequences = []
    for j in J:
        for a in A_j[j]:
            sequences.append((j, a))
    sequence_index = {s: idx for idx, s in enumerate(sequences)}
    
    # Build matrix representation of strategy polytope
    F = [] # Matrix of shape |J| x |sequences| (each row is represented as a dict)
    f = [] # Vector of shape |J|
    
    for j in J:
        row = defaultdict(float)
        for a in A_j[j]:
            row[sequence_index[(j, a)]] += 1.0
            
        parent_sequence = p_j[j]
        if parent_sequence is None:
            f.append(1.0)
        else:
            row[sequence_index[parent_sequence]] -= 1.0
            f.append(0.0)
            
        F.append(row)
    return sequences, F, f, J, A_j, p_j


def build_payoff_matrix(game, S1, S2):
    
    # Build payoff matrix of shape |S1| x |S2| in terms of player 1's utility (only store non-zero entres using a dict)
    A = defaultdict(float)
    r_idx = {s: i for i, s in enumerate(S1)}
    c_idx = {s: j for j, s in enumerate(S2)}

    for leaf_node in game["utility_pl1"]:
        s1 = leaf_node["sequence_pl1"]
        s2 = leaf_node["sequence_pl2"]
        value = float(leaf_node["value"])
        
        if s1 in r_idx and s2 in c_idx:
            A[(r_idx[s1], c_idx[s2])] += value
    return A


def build_all_matrices(game):
    
    S1, F1, f1, J1, A1_j, p1_j = build_strategy_polytope(game["decision_problem_pl1"])
    S2, F2, f2, J2, A2_j, p2_j = build_strategy_polytope(game["decision_problem_pl2"])
    A = build_payoff_matrix(game, S1, S2)
    return S1, F1, f1, S2, F2, f2, A, J1, A1_j, p1_j, J2, A2_j, p2_j


def solve_problem_2_3_with_k(player, k, S1, F1, f1, S2, F2, f2, A, J1, A1_j, p1_j, J2, A2_j, p2_j):
    
    n1, n2 = len(S1), len(S2)
    m1, m2 = len(F1), len(F2)
    
    m = gurobi.Model("game_value_pl" + str(player) + "_" + str(k))
    
    if player == 1:
        
        # Player 1 free variables
        x = m.addVars(n1, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="x")
        v = m.addVars(m2, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="v")
        z = m.addVars(n1, vtype=GRB.BINARY, name="z")
        
        # Constraint F1.x = f1
        for r_idx, row in enumerate(F1):
            expr = gurobi.LinExpr()
            for c_idx, coeff in row.items():
                expr.addTerms(coeff, x[c_idx])
            m.addConstr(expr == f1[r_idx], name=f"F1.x[{r_idx}]")
            
        # Constraint A^T.x - F2^T.v >= 0
        for j in range(n2):
            lhs = gurobi.LinExpr()
            
            for (r_idx, c_idx), value in A.items():
                if c_idx == j:
                    lhs.addTerms(value, x[r_idx])
                    
            for r_idx, row in enumerate(F2):
                coeff = row.get(j, 0.0)
                if coeff:
                    lhs.addTerms(-coeff, v[r_idx])
                    
            rhs = 0.0
            m.addConstr(lhs >= rhs, name=f"A^T.x-F2^T.v[{j}]")
            
        # Constraint x[(j,a)] >= z[(j,a)] for p1_j[j] == None
        for j in J1:
            if p1_j[j] is None:
                for a in A1_j[j]:
                    seq_idx = S1.index((j, a))
                    m.addConstr(x[seq_idx] >= z[seq_idx], name=f"x[(j,a)]>=z[(j,a)][{(j,a)}]")
                    
        # Constraint x[(j,a)] >= x[p_j] + z[(j,a)] - 1 for p1_j[j] != None
        for j in J1:
            if p1_j[j] is not None:
                parent_idx = S1.index(p1_j[j])
                for a in A1_j[j]:
                    seq_idx = S1.index((j, a))
                    m.addConstr(x[seq_idx] - x[parent_idx] - z[seq_idx] >= -1.0, name=f"x[(j,a)]>=x[p_j]+z[(j,a)]-1[{(j,a)}]")
                    
        # Constraint sum_a z[(j,a)] <= 1
        for j in J1:
            seqs = [S1.index((j, a)) for a in A1_j[j]]
            m.addConstr(gurobi.quicksum(z[i] for i in seqs) <= 1)
            
        # Constraint sum_j sum_a z[(j,a)] >= k
        m.addConstr(gurobi.quicksum(z[i] for i in range(n1)) >= k)
        
        # Objective max f2^T.v
        obj = gurobi.LinExpr()
        for r_idx in range(m2):
            obj.addTerms(f2[r_idx], v[r_idx])
            
        m.setObjective(obj, sense=GRB.MAXIMIZE)
        
    else:
        
        # Player 2 free variables
        y = m.addVars(n2, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="y")
        v = m.addVars(m1, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="v")
        z = m.addVars(n2, vtype=GRB.BINARY, name="z")
        
        # Constraint F2.y = f2
        for r_idx, row in enumerate(F2):
            expr = gurobi.LinExpr()
            for c_idx, coeff in row.items():
                expr.addTerms(coeff, y[c_idx])
            m.addConstr(expr == f2[r_idx], name=f"F2.y[{r_idx}]")

        # Constraint -A.y - F1^T.v >= 0
        for i in range(n1):
            lhs = gurobi.LinExpr()
            
            for (r_idx, c_idx), value in A.items():
                if r_idx == i:
                    lhs.addTerms(-value, y[c_idx])
                    
            for r_idx, row in enumerate(F1):
                coeff = row.get(i, 0.0)
                if coeff:
                    lhs.addTerms(-coeff, v[r_idx])
                
            rhs = 0.0        
            m.addConstr(lhs >= rhs, name=f"-A.y-F1^T.v[{i}]")
            
        # Constraint y[(j,a)] >= z[(j,a)] for p2_j[j] == None
        for j in J2:
            if p2_j[j] is None:
                for a in A2_j[j]:
                    seq_idx = S2.index((j, a))
                    m.addConstr(y[seq_idx] >= z[seq_idx], name=f"y[(j,a)]>=z[(j,a)][{(j,a)}]")
                    
        # Constraint y[(j,a)] >= y[p_j] + z[(j,a)] - 1 for p2_j[j] != None
        for j in J2:
            if p2_j[j] is not None:
                parent_idx = S2.index(p2_j[j])
                for a in A2_j[j]:
                    seq_idx = S2.index((j, a))
                    m.addConstr(y[seq_idx] - y[parent_idx] - z[seq_idx] >= -1.0, name=f"y[(j,a)]>=y[p_j]+z[(j,a)]-1[{(j,a)}]")
                    
        # Constraint sum_a z[(j,a)] <= 1
        for j in J2:
            seqs = [S2.index((j, a)) for a in A2_j[j]]
            m.addConstr(gurobi.quicksum(z[i] for i in seqs) <= 1)
            
        # Constraint sum_j sum_a z[(j,a)] >= k
        m.addConstr(gurobi.quicksum(z[i] for i in range(n2)) >= k)
        
        # Objective max f1^T.v
        obj = gurobi.LinExpr()
        for r_idx in range(m1):
            obj.addTerms(f1[r_idx], v[r_idx])
            
        m.setObjective(obj, sense=GRB.MAXIMIZE)
    
    m.optimize()
    return m


def plot_objectives(player, k_values, obj_values):

    color = 'blue' if player == 1 else 'red'
    plt.figure()
    plt.plot(k_values, obj_values, marker='o', color=color, label=f"Player {player}")
    plt.title(f"Player {player} Objective vs k")
    plt.xlabel("k (Number of Deterministic Decision Points)")
    plt.ylabel("Objective Value")
    plt.grid(True)
    plt.legend()
    plt.show()


def solve_problem_2_1(game):
    
    S1, F1, f1, S2, F2, f2, A, _, _, _, _, _, _ = build_all_matrices(game)
    n1, n2 = len(S1), len(S2)
    m1, m2 = len(F1), len(F2)

    for player in [1, 2]:
        
        m = gurobi.Model("game_value_pl" + str(player))

        if player == 1:

            # Player 1 free variables
            # Reference: https://docs.gurobi.com/projects/optimizer/en/current/reference/python/model.html
            x = m.addVars(n1, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="x")
            v = m.addVars(m2, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="v")

            # Constraint F1.x = f1
            # Reference: https://docs.gurobi.com/projects/optimizer/en/current/reference/python/linexpr.html
            for r_idx, row in enumerate(F1):
                expr = gurobi.LinExpr()
                for c_idx, coeff in row.items():
                    expr.addTerms(coeff, x[c_idx])
                m.addConstr(expr == f1[r_idx], name=f"F1.x[{r_idx}]")

            # Constraint A^T.x - F2^T.v >= 0
            for j in range(n2):
                lhs = gurobi.LinExpr()
                
                for (r_idx, c_idx), value in A.items():
                    if c_idx == j:
                        lhs.addTerms(value, x[r_idx])
                        
                for r_idx, row in enumerate(F2):
                    coeff = row.get(j, 0.0)
                    if coeff:
                        lhs.addTerms(-coeff, v[r_idx])
                        
                rhs = 0.0
                m.addConstr(lhs >= rhs, name=f"A^T.x-F2^T.v[{j}]")

            # Objective max f2^T.v
            obj = gurobi.LinExpr()
            for r_idx in range(m2):
                obj.addTerms(f2[r_idx], v[r_idx])
                
            m.setObjective(obj, sense=GRB.MAXIMIZE)

        else:
            # Player 2 free variables
            y = m.addVars(n2, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="y")
            v = m.addVars(m1, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="v")

            # Constraint F2.y = f2
            for r_idx, row in enumerate(F2):
                expr = gurobi.LinExpr()
                for c_idx, coeff in row.items():
                    expr.addTerms(coeff, y[c_idx])
                m.addConstr(expr == f2[r_idx], name=f"F2.y[{r_idx}]")

            # Constraint -A.y - F1^T.v >= 0
            for i in range(n1):
                lhs = gurobi.LinExpr()
                
                for (r_idx, c_idx), value in A.items():
                    if r_idx == i:
                        lhs.addTerms(-value, y[c_idx])
                        
                for r_idx, row in enumerate(F1):
                    coeff = row.get(i, 0.0)
                    if coeff:
                        lhs.addTerms(-coeff, v[r_idx])
                 
                rhs = 0.0        
                m.addConstr(lhs >= rhs, name=f"-A.y-F1^T.v[{i}]")

            # Objective max f1^T.v
            obj = gurobi.LinExpr()
            for r_idx in range(m1):
                obj.addTerms(f1[r_idx], v[r_idx])
                
            m.setObjective(obj, sense=GRB.MAXIMIZE)

        m.optimize()

        if m.Status == GRB.OPTIMAL:
            print(f"Player {player} Objective:", m.getAttr(GRB.Attr.ObjVal))
            print(f"--------------------------------------------------------------------")


def solve_problem_2_2(game):
    
    S1, F1, f1, S2, F2, f2, A, _, _, _, _, _, _ = build_all_matrices(game)
    n1, n2 = len(S1), len(S2)
    m1, m2 = len(F1), len(F2)

    for player in [1, 2]:
        
        m = gurobi.Model("game_value_pl" + str(player))

        if player == 1:

            # Player 1 free variables
            # Reference: https://docs.gurobi.com/projects/optimizer/en/current/reference/python/model.html
            x = m.addVars(n1, vtype=GRB.BINARY, name="x")
            v = m.addVars(m2, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="v")

            # Constraint F1.x = f1
            # Reference: https://docs.gurobi.com/projects/optimizer/en/current/reference/python/linexpr.html
            for r_idx, row in enumerate(F1):
                expr = gurobi.LinExpr()
                for c_idx, coeff in row.items():
                    expr.addTerms(coeff, x[c_idx])
                m.addConstr(expr == f1[r_idx], name=f"F1.x[{r_idx}]")

            # Constraint A^T.x - F2^T.v >= 0
            for j in range(n2):
                lhs = gurobi.LinExpr()
                
                for (r_idx, c_idx), value in A.items():
                    if c_idx == j:
                        lhs.addTerms(value, x[r_idx])
                        
                for r_idx, row in enumerate(F2):
                    coeff = row.get(j, 0.0)
                    if coeff:
                        lhs.addTerms(-coeff, v[r_idx])
                        
                rhs = 0.0
                m.addConstr(lhs >= rhs, name=f"A^T.x-F2^T.v[{j}]")

            # Objective max f2^T.v
            obj = gurobi.LinExpr()
            for r_idx in range(m2):
                obj.addTerms(f2[r_idx], v[r_idx])
                
            m.setObjective(obj, sense=GRB.MAXIMIZE)

        else:
            # Player 2 free variables
            y = m.addVars(n2, vtype=GRB.BINARY, name="y")
            v = m.addVars(m1, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="v")

            # Constraint F2.y = f2
            for r_idx, row in enumerate(F2):
                expr = gurobi.LinExpr()
                for c_idx, coeff in row.items():
                    expr.addTerms(coeff, y[c_idx])
                m.addConstr(expr == f2[r_idx], name=f"F2.y[{r_idx}]")

            # Constraint -A.y - F1^T.v >= 0
            for i in range(n1):
                lhs = gurobi.LinExpr()
                
                for (r_idx, c_idx), value in A.items():
                    if r_idx == i:
                        lhs.addTerms(-value, y[c_idx])
                        
                for r_idx, row in enumerate(F1):
                    coeff = row.get(i, 0.0)
                    if coeff:
                        lhs.addTerms(-coeff, v[r_idx])
                 
                rhs = 0.0        
                m.addConstr(lhs >= rhs, name=f"-A.y-F1^T.v[{i}]")

            # Objective max f1^T.v
            obj = gurobi.LinExpr()
            for r_idx in range(m1):
                obj.addTerms(f1[r_idx], v[r_idx])
                
            m.setObjective(obj, sense=GRB.MAXIMIZE)

        m.optimize()

        if m.Status == GRB.OPTIMAL:
            print(f"Player {player} Objective:", m.getAttr(GRB.Attr.ObjVal))
            print(f"--------------------------------------------------------------------")


def solve_problem_2_3(game):
    
    S1, F1, f1, S2, F2, f2, A, J1, A1_j, p1_j, J2, A2_j, p2_j = build_all_matrices(game)
    
    for player in [1, 2]:
        Ji = J1 if player == 1 else J2
        max_k = len(Ji)
        k_values = []
        obj_values = []
        
        for k in range(0, max_k + 1):
            m = solve_problem_2_3_with_k(player, k, S1, F1, f1, S2, F2, f2, A, J1, A1_j, p1_j, J2, A2_j, p2_j)
            
            if m.Status == GRB.OPTIMAL:
                print(f"Player {player} Objective with k={k}:", m.getAttr(GRB.Attr.ObjVal))
                k_values.append(k)
                obj_values.append(m.getAttr(GRB.Attr.ObjVal))
                
            else:
                print(f"Player {player} Objective with k={k} is infeasible.")
                break
        
        if obj_values:
            plot_objectives(player, k_values, obj_values)
        print(f"--------------------------------------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='HW2 Problem 2 (Deterministic strategies)')
    parser.add_argument("--game", help="Path to game file", required=True)
    parser.add_argument(
        "--problem", choices=["2.1", "2.2", "2.3"], required=True)

    args = parser.parse_args()
    print("Reading game path %s..." % args.game)

    game = json.load(open(args.game))

    # Convert all sequences from lists to tuples
    # tfsdp = Terminal-Free Sequence-Form Decision Problem
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

    if args.problem == "2.1":
        solve_problem_2_1(game)
    elif args.problem == "2.2":
        solve_problem_2_2(game)
    else:
        assert args.problem == "2.3"
        solve_problem_2_3(game)