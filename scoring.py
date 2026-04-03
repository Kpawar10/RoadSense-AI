def calculate_score(speed, acceleration, braking, turn_rate):
    score = 100

    # Penalties
    if speed > 70: 
        score -= (speed - 70) * 0.5

    if acceleration > 3:
        score -= (acceleration - 3) * 5

    if braking < -3:
        score -= abs(braking + 3) * 5

    if turn_rate > 20:
        score -= (turn_rate - 20) * 2

    return max(0, min(100, score))