import numpy as np


N_DAYS = 100
MAX_OCCUPANCY = 300
MIN_OCCUPANCY = 125

def main():
    from data_loader import load_data, process_data, find_best_result
    
    data = load_data('../input/family_data.csv')
    family_size_dict, choice_dict = process_data(data)
    days = list(range(N_DAYS,0,-1))
    
    def cost_function(prediction):
        penalty = 0

        # We'll use this to count the number of people scheduled each day
        daily_occupancy = {k:0 for k in days}

        # Looping over each family; d is the day for each family f
        for f, d in enumerate(prediction):

            # Using our lookup dictionaries to make simpler variable names
            n = family_size_dict[f]
            choice_0 = choice_dict['choice_0'][f]
            choice_1 = choice_dict['choice_1'][f]
            choice_2 = choice_dict['choice_2'][f]
            choice_3 = choice_dict['choice_3'][f]
            choice_4 = choice_dict['choice_4'][f]
            choice_5 = choice_dict['choice_5'][f]
            choice_6 = choice_dict['choice_6'][f]
            choice_7 = choice_dict['choice_7'][f]
            choice_8 = choice_dict['choice_8'][f]
            choice_9 = choice_dict['choice_9'][f]

            # add the family member count to the daily occupancy
            daily_occupancy[d] += n

            # Calculate the penalty for not getting top preference
            if d == choice_0:
                penalty += 0
            elif d == choice_1:
                penalty += 50
            elif d == choice_2:
                penalty += 50 + 9 * n
            elif d == choice_3:
                penalty += 100 + 9 * n
            elif d == choice_4:
                penalty += 200 + 9 * n
            elif d == choice_5:
                penalty += 200 + 18 * n
            elif d == choice_6:
                penalty += 300 + 18 * n
            elif d == choice_7:
                penalty += 300 + 36 * n
            elif d == choice_8:
                penalty += 400 + 36 * n
            elif d == choice_9:
                penalty += 500 + 36 * n + 199 * n
            else:
                penalty += 500 + 36 * n + 398 * n

        # for each date, check total occupancy
        #  (using soft constraints instead of hard constraints)
        for _, v in daily_occupancy.items():
            if (v > MAX_OCCUPANCY) or (v < MIN_OCCUPANCY):
                penalty += 100000000

        # Calculate the accounting cost
        # The first day (day 100) is treated special
        accounting_cost = (daily_occupancy[days[0]]-125.0) / 400.0 * daily_occupancy[days[0]]**(0.5)
        # using the max function because the soft constraints might allow occupancy to dip below 125
        accounting_cost = max(0, accounting_cost)

        # Loop over the rest of the days, keeping track of previous count
        yesterday_count = daily_occupancy[days[0]]
        for day in days[1:]:
            today_count = daily_occupancy[day]
            diff = abs(today_count - yesterday_count)
            accounting_cost += max(0, (daily_occupancy[day]-125.0) / 400.0 * daily_occupancy[day]**(0.5 + diff / 50.0))
            yesterday_count = today_count

        penalty += accounting_cost

        return penalty    
    
    best_file, best_score = find_best_result('../output/')
    if best_file is not None:
        print(f'Best file found: {best_file}')
        submission = load_data(f'../output/{best_file}')
    else:   
        submission = load_data('../output/sample_submission.csv')
    best = submission['assigned_day'].tolist()
    start_score = cost_function(best)
    new = best.copy()
    # loop over each family
    for fam_id, _ in enumerate(best):
        # loop over each family choice
        for pick in range(10):
            day = choice_dict[f'choice_{pick}'][fam_id]
            temp = new.copy()
            temp[fam_id] = day # add in the new pick
            if cost_function(temp) < start_score:
                new = temp.copy()
                start_score = cost_function(new)

    submission['assigned_day'] = new
    score = cost_function(new)
    if(score < best_score):
        submission.to_csv(f'../output/submission_{score}.csv')
    print(f'Score: {score}')


if __name__ == "__main__":
    main()
