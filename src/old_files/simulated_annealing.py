from data_loader import load_data, process_data, find_best_result
import random, math, json, time

N_DAYS = 100
MAX_OCCUPANCY = 300
MIN_OCCUPANCY = 125

# Simulated Annealing parameters
CALIBRATION_SAMPLES = 1000
TARGET_ACCEPTANCE = 0.25
MAX_ITERATIONS = 10000000
MAX_RUNNING_TIME = 1800
NUM_ROUNDS = 20
PARAMS_FILE = '../output/sa_params.json'

def build_choices(choice_dict, num_families):
    return {fid: [choice_dict[f'choice_{i}'][fid] for i in range(10)] for fid in range(num_families)}

def preference_cost(family_size, choice, assigned_day):
    if assigned_day == choice[0]:
        return 0
    elif assigned_day == choice[1]:
        return 50
    elif assigned_day == choice[2]:
        return 50 + 9 * family_size
    elif assigned_day == choice[3]:
        return 100 + 9 * family_size
    elif assigned_day == choice[4]:
        return 200 + 9 * family_size
    elif assigned_day == choice[5]:
        return 200 + 18 * family_size
    elif assigned_day == choice[6]:
        return 300 + 18 * family_size
    elif assigned_day == choice[7]:
        return 300 + 36 * family_size
    elif assigned_day == choice[8]:
        return 400 + 36 * family_size
    elif assigned_day == choice[9]:
        return 500 + 36 * family_size + 199 * family_size
    else:
        return 500 + 36 * family_size + 398 * family_size

def generate_cost_dict(family_size_dict, choices, assignment):
    cost_dict = {}
    for family_id in family_size_dict.keys():
        cost_dict[family_id] = preference_cost(family_size_dict[family_id],
                                               choices[family_id],
                                               assignment[family_id])
    return cost_dict
    
def accounting_cost_for_day(day, daily_occupancy):
    n_d = daily_occupancy[day]
    if(n_d < MIN_OCCUPANCY or n_d > MAX_OCCUPANCY):
        return 1000000 # must change to hard constrains later
    if day == N_DAYS:
        return max(0, (n_d - 125.0) / 400.0 * n_d ** 0.5)
    n_next = daily_occupancy[day + 1]
    diff = abs(n_d - n_next)
    return max(0, (n_d - 125.0) / 400.0 * n_d ** (0.5 + diff / 50.0))

def delta_cost(family_id, new_day, family_size_dict, choices, assignment, cost_dict, daily_occupancy):
    old_day = assignment[family_id]
    n = family_size_dict[family_id]

    old_pref = cost_dict[family_id]
    new_pref = preference_cost(n, choices[family_id], new_day)
    pref_delta = new_pref - old_pref

    affected = set()
    for d in (old_day, old_day - 1, new_day, new_day - 1):
        if 1 <= d <= N_DAYS:
            affected.add(d)

    acc_before = sum(accounting_cost_for_day(d, daily_occupancy) for d in affected)

    daily_occupancy[old_day] -= n
    daily_occupancy[new_day] += n

    acc_after = sum(accounting_cost_for_day(d, daily_occupancy) for d in affected)

    daily_occupancy[old_day] += n
    daily_occupancy[new_day] -= n

    return pref_delta + (acc_after - acc_before), new_pref
    

def total_score(cost_dict, daily_occupancy):
    return sum(cost_dict.values()) + sum(accounting_cost_for_day(d, daily_occupancy) for d in range(1, N_DAYS + 1))

def apply_move(family_id, new_day, new_pref, family_size_dict, assignment, cost_dict, daily_occupancy):
    n = family_size_dict[family_id]
    old_day = assignment[family_id]
    daily_occupancy[old_day] -= n
    daily_occupancy[new_day] += n
    assignment[family_id] = new_day
    cost_dict[family_id] = new_pref

def greedy(family_size_dict, choices, assignment, cost_dict, daily_occupancy):
    improved = True
    while improved:
        improved = False
        for family_id in range(len(family_size_dict)):
            for pick in range(10):
                new_day = choices[family_id][pick]
                if new_day == assignment[family_id]:
                    continue
                delta, new_pref = delta_cost(family_id, new_day, family_size_dict, choices, assignment, cost_dict, daily_occupancy)
                if delta < 0:
                    apply_move(family_id, new_day, new_pref, family_size_dict, assignment, cost_dict, daily_occupancy)
                    improved = True

def is_feasible_move(family_id, new_day, family_size_dict, assignment, daily_occupancy):
    n = family_size_dict[family_id]
    old_day = assignment[family_id]
    if daily_occupancy[old_day] - n < MIN_OCCUPANCY:
        return False
    if daily_occupancy[new_day] + n > MAX_OCCUPANCY:
        return False
    return True

def calibrate_temperature(family_size_dict, choices, assignment, cost_dict, daily_occupancy,
                          num_samples, target_acceptance):
    uphill_deltas = []
    num_families = len(family_size_dict)
    while len(uphill_deltas) < num_samples:
        family_id = random.randint(0, num_families - 1)
        new_day = random.choice(choices[family_id])
        if new_day == assignment[family_id]:
            continue
        if not is_feasible_move(family_id, new_day, family_size_dict, assignment, daily_occupancy):
            continue
        delta, _ = delta_cost(family_id, new_day, family_size_dict, choices, assignment, cost_dict, daily_occupancy)
        if delta > 0:
            uphill_deltas.append(delta)

    uphill_deltas.sort()
    # pick the delta at the target_acceptance percentile
    # at T_initial, we want target_acceptance fraction of uphill moves accepted
    # exp(-delta / T) = target_acceptance => T = -delta / ln(target_acceptance)
    idx = int(len(uphill_deltas) * target_acceptance)
    median_delta = uphill_deltas[idx]
    t_initial = -median_delta / math.log(target_acceptance)

    print(f'Calibrated T_initial: {t_initial:.2f} (median uphill delta at {target_acceptance*100:.0f}%: {median_delta:.2f})')
    return t_initial

def simulated_annealing(family_size_dict, choices, assignment, cost_dict, daily_occupancy,
                        max_iterations):
    num_families = len(family_size_dict)

    t_initial = calibrate_temperature(family_size_dict, choices, assignment, cost_dict, daily_occupancy,
                                      CALIBRATION_SAMPLES, TARGET_ACCEPTANCE)
    
    # cooling rate so that final temp is 1% of initial
    # alpha = (0.01 * t_initial / t_initial) ** (1.0 / max_iterations)
    alpha = 0.01 ** (1.0 / max_iterations)
    print(f'Cooling rate (alpha): {alpha:.10f}')

    t = t_initial
    current_score = total_score(cost_dict, daily_occupancy)
    best_score = current_score
    best_assignment = assignment[:]
    best_cost_dict = cost_dict.copy()
    best_occupancy = daily_occupancy.copy()

    for iteration in range(max_iterations):
        family_id = random.randint(0, num_families - 1)
        new_day = random.choice(choices[family_id])
        if new_day == assignment[family_id]:
            continue
        if not is_feasible_move(family_id, new_day, family_size_dict, assignment, daily_occupancy):
            continue
        delta, new_pref = delta_cost(family_id, new_day, family_size_dict, choices, assignment, cost_dict, daily_occupancy)
        if delta < 0 or random.random() < math.exp(-delta / t):
            apply_move(family_id, new_day, new_pref, family_size_dict, assignment, cost_dict, daily_occupancy)
            current_score += delta
            if current_score < best_score:
                best_score = current_score
                best_assignment = assignment[:]
                best_cost_dict = cost_dict.copy()
                best_occupancy = daily_occupancy.copy()

        t *= alpha

        if iteration % 500000 == 0:
            print(f'Iter {iteration:>10,} | T: {t:.4f} | Current: {current_score:.2f} | Best: {best_score:.2f}')
            
    print(f'Iter {max_iterations:>10,} | T: {t:.4f} | Current: {current_score:.2f} | Best: {best_score:.2f}')
    
    assignment[:] = best_assignment
    cost_dict.update(best_cost_dict)
    daily_occupancy.update(best_occupancy)

    params = {
        't_initial': t_initial,
        'alpha': alpha,
        'max_iterations': max_iterations,
        'target_acceptance': TARGET_ACCEPTANCE,
        'best_score': best_score,
        'start_score': total_score(cost_dict, daily_occupancy),
    }
    
    with open(PARAMS_FILE, 'w') as f:
        json.dump(params, f, indent=2)
    print(f'Saved SA params to {PARAMS_FILE}')

def main():
    data = load_data('../input/family_data.csv')
    family_size_dict, choice_dict = process_data(data)

    best_file, _ = find_best_result('../output/')
    if best_file is not None:
        print(f'Best file found: {best_file}')
        submission = load_data(f'../output/{best_file}')
    else:
        submission = load_data('../output/sample_submission.csv')
    assignment = submission['assigned_day'].tolist()
    daily_occupancy = {}
    for index, day in enumerate(assignment):
        if day not in daily_occupancy:
            daily_occupancy[day] = 0
        daily_occupancy[day] += family_size_dict[index]

    choices = build_choices(choice_dict, len(family_size_dict))
    cost_dict = generate_cost_dict(family_size_dict, choices, assignment)

    greedy(family_size_dict, choices, assignment, cost_dict, daily_occupancy)
    print(f'After greedy: {total_score(cost_dict, daily_occupancy):.2f}')

    elapsed = 0
    rounds = 0
    
    while elapsed < MAX_RUNNING_TIME:
        print(f'\n=== Round {rounds + 1} - {elapsed:.2f}s ===')
        start_time = time.time()
        simulated_annealing(family_size_dict, choices, assignment, cost_dict, daily_occupancy,
                            MAX_ITERATIONS)
        end_time = time.time()
        elapsed += end_time - start_time
        rounds += 1
        greedy(family_size_dict, choices, assignment, cost_dict, daily_occupancy)
        score = total_score(cost_dict, daily_occupancy)
        print(f'After greedy cleanup: {score:.2f}')

    submission['assigned_day'] = assignment
    score = total_score(cost_dict, daily_occupancy)
    print(f'\nTotal elapsed time: {elapsed:.2f}s over {rounds} rounds')
    print(f'\nFinal score: {score}')

    best_file, best_score = find_best_result('../output/')
    if score < best_score:
        submission.to_csv(f'../output/submission_{score}.csv')
        print(f'Saved new best: {score}')

if __name__ == "__main__":
    main()