import re
from typing import Any

def parse_response(response: str):
    """Parse the response.

    Returns a parsed plan in a list of (city, stay_days) tuples.

    Args:
        response: Raw response from the model.

    Returns:
        Structured plan after parsing.
    """
    pattern_visit = r'\d+-\d+'
    pattern_flight = r'.*Day (\d+).*from (\w+) to (\w+)'
    pattern_days = r'European cities for (\d+) days'

    days, flights, flight_days = [], [], []
    total_days = None
    for piece in response.split('\n'):
        days_match = re.findall(pattern_days, piece)
        if days_match:
            total_days = int(days_match[0])

        visit_match = re.findall(pattern_visit, piece)
        if visit_match:
            days.append(visit_match[0])
            end_day = int(visit_match[0].split('-')[1])
            # Reach the end of the plan, stop to avoid parsing alternative plans.
            if end_day == total_days:
                break
        flight_match = re.findall(pattern_flight, piece)
        if flight_match:
            flights.append(flight_match[0])

    visit_cities, parsed_plan = [], []
    for flight_day, begin_city, end_city in flights:
        flight_days.append(int(flight_day))
        if not visit_cities:
            visit_cities.append(begin_city)
            visit_cities.append(end_city)
        else:
            visit_cities.append(end_city)

    if not days or not flights or not visit_cities:
        return []
    last_day = int(days[-1].split('-')[1])
    flight_days = [1] + flight_days + [last_day]
    for i, visit_city in enumerate(visit_cities):
        city_stay = flight_days[i + 1] - flight_days[i] + 1
        parsed_plan.append((visit_city, city_stay))

    return parsed_plan


def compute_example_score(cities: str, durations: str, parsed_plan: list[Any]):
    """Compute the exact-match accuracy.

    Compute the example-level exact_match score (0/1) given the parsed plan
    and the ground truth in the format of durations and cities.

    Args:
        cities: The cities in the plan in the format of "city1**city2**city3".
        durations: The durations of the stay in each city in the format of
        "1**2**3".
        parsed_plan: The parsed plan from the response.

    Returns:
        Exact-match accuracy of 0 (mismatched) or 1 (matched).
    """

    stays = [x for x in cities.split('**') if x]
    days = [int(x) for x in durations.split('**') if x]
    num_stays = min(len(stays), len(parsed_plan))
    num_match = 0
    for i in range(num_stays):
        if stays[i] == parsed_plan[i][0] and days[i] == parsed_plan[i][1]:
            num_match += 1
        else:
            break
    hard_score = 0.0 if num_match / len(stays) < 1.0 else 1.0
    return hard_score


def compute_score(
    cities: list[str], durations: list[str], responses: list[str]
):
    """Compute the sample-level exact-match accuracy.

    Args:
        cities: List of cities in the plan in the format of "city1**city2**city3".
        durations: List of durations of the stay in each city in the format of
        "1**2**3".
        responses: The raw responses from the model.

    Returns:
        Exact-match score at the sample level.
    """
    parsed_plans = [parse_response(response) for response in responses]
    hard_scores = [
        compute_example_score(city, duration, parsed_plan)
        for city, duration, parsed_plan in zip(cities, durations, parsed_plans)
    ]
    print([i for i, j in enumerate(hard_scores) if j == 1.0])
    hard_acc = sum(hard_scores) / len(hard_scores)
    return hard_acc

def trip_metric(data, preds):
    cities, durations, responses = [], [], []
    sample_count = 0
    for item, pred in zip(data.values(), preds):
        cities.append(item['cities'])
        durations.append(item['durations'])
        responses.append(pred)
        sample_count += 1

    hard_acc = compute_score(cities, durations, responses)
    print(f'EM Accuracy of {sample_count} samples: {hard_acc}')

