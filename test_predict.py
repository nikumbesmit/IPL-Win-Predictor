from src.predict import predict_win_probability

result = predict_win_probability(
    batting_team="Mumbai Indians",
    bowling_team="Chennai Super Kings",
    city="Mumbai",
    target=180,
    score=120,
    overs_completed=14.2,
    wickets_out=4
)

print(result)