import re
from collections import Counter

def cd_metric(inputs, preds):
    def check_eq(left_str, right_str):
        left_matches = re.match(r'(\d+)([+\-*/])(\d+)', left_str)
        if left_matches:
            return eval(left_str) == float(right_str)
        else:
            return False

    cor = 0

    for query, pred in zip(inputs, preds):
        subequations = pred.split(',')  # sub-equations
        match = True
        query_numbers = Counter(query.split(',')[:-1])
        for subeq in subequations:
            try:
                left, right = subeq.split('=')
                match &= check_eq(left, right)
                left_side_numbers = re.findall(r'(\d+)(?=[+-/*=])', subeq)
                query_numbers.subtract(left_side_numbers)
                query_numbers.update({right:1})
            except:
                match = False
            if not match:
                break

        answer = query.split(',')[-1]
        pred_ans = pred.split('=')[-1]

        query_numbers.subtract({query.split(',')[-1]: 1})
        numbers_match = all(value == 0 for value in query_numbers.values())
        # if not numbers_match:
        #     print(query +"\t" + label + "\t" + pred)
        cor += (match and (answer == pred_ans) and numbers_match)
        # cor += (match and (answer == pred_ans))

    return cor/len(preds)
