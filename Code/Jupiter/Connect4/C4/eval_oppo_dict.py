# C4.eval_oppo_dict.py
# 

#final evaluation
EVALUATION_OPPONENTS = {
    "Random": 200,
    "Leftmost": 100,
    "Center": 200,
    "Lookahead-1": 100,
    "Lookahead-2": 100,
    "Lookahead-3": 100,
    "Lookahead-4": 100,
    "Lookahead-5": 50,
    "Lookahead-6": 24,
    "Lookahead-7": 10,
    "Lookahead-9": 6,
    "Lookahead-11": 4,
    "Lookahead-13": 4,
}

#online training evals
EVAL_CFG = {
    "Random": 200, 
    "Lookahead-1": 100,
    "Lookahead-2": 10,
    "Lookahead-3": 30,
    "Lookahead-4": 10,
    "Lookahead-5": 10,
    "Lookahead-6": 4,
    "Lookahead-7": 4,
    } 

OPENING_NOISE_K = {0:0.85, 1:0.10, 2:0.05}