import pickle
import pandas as pd


MODEL_PATH = "artifacts/model.pkl"
METADATA_PATH = "artifacts/metadata.pkl"


def load_artifacts(model_path=MODEL_PATH, metadata_path=METADATA_PATH):
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    return model, metadata


def overs_to_balls(overs):

    whole_overs = int(overs)
    balls_part = round((overs - whole_overs) * 10)

    if balls_part > 5:
        raise ValueError("Invalid overs format. Decimal part must be between 0 and 5.")

    return whole_overs * 6 + balls_part


def build_match_state(
    batting_team,
    bowling_team,
    city,
    target,
    score,
    overs_completed,
    wickets_out
):


    if batting_team == bowling_team:
        raise ValueError("Batting team and bowling team cannot be the same.")

    if overs_completed < 0 or overs_completed >= 20:
        raise ValueError("Overs completed must be between 0 and less than 20.")

    if wickets_out < 0 or wickets_out > 9:
        raise ValueError("Wickets out must be between 0 and 9.")

    if score < 0:
        raise ValueError("Score cannot be negative.")

    if target <= 0:
        raise ValueError("Target must be positive.")

    if score > target:
        raise ValueError("Current score cannot be greater than target for an in-progress chase.")

    balls_bowled = overs_to_balls(overs_completed)
    balls_left = 120 - balls_bowled

    if balls_left <= 0:
        raise ValueError("No balls left. Please enter overs less than 20.")

    runs_left = target - score
    wickets_left = 10 - wickets_out

    # Keep the same logic style used in training
    crr = (score * 6) / balls_bowled if balls_bowled > 0 else 0
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

    input_df = pd.DataFrame([{
        "batting_team": batting_team,
        "bowling_team": bowling_team,
        "city": city,
        "runs_left": runs_left,
        "balls_left": balls_left,
        "wickets_left": wickets_left,
        "target": target,
        "crr": crr,
        "rrr": rrr
    }])

    return input_df


def predict_win_probability(
    batting_team,
    bowling_team,
    city,
    target,
    score,
    overs_completed,
    wickets_out
):

    model, metadata = load_artifacts()

    input_df = build_match_state(
        batting_team=batting_team,
        bowling_team=bowling_team,
        city=city,
        target=target,
        score=score,
        overs_completed=overs_completed,
        wickets_out=wickets_out
    )

    proba = model.predict_proba(input_df)[0]

    batting_team_win_prob = float(proba[1])
    bowling_team_win_prob = float(proba[0])

    return {
        "batting_team_win_probability": round(batting_team_win_prob * 100, 2),
        "bowling_team_win_probability": round(bowling_team_win_prob * 100, 2),
        "input_df": input_df
    }


def get_metadata():
    _, metadata = load_artifacts()
    return metadata