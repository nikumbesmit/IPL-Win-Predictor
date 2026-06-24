import streamlit as st
from src.predict import predict_win_probability, get_metadata


st.set_page_config(
    page_title="IPL Win Predictor",
    page_icon="🏏",
    layout="wide"
)


metadata = get_metadata()
teams = sorted(metadata["teams"])
cities = sorted(metadata["cities"])

st.markdown("""
    <style>
        .main-title {
            font-size: 42px;
            font-weight: 700;
            margin-bottom: 0;
        }
        .subtitle {
            font-size: 18px;
            color: #6b7280;
            margin-top: 0;
            margin-bottom: 25px;
        }
        .card {
            padding: 20px;
            border-radius: 16px;
            background-color: #111827;
            color: white;
            margin-bottom: 20px;
        }
        .metric-title {
            font-size: 16px;
            color: #d1d5db;
        }
        .metric-value {
            font-size: 28px;
            font-weight: 700;
        }
        .small-box {
            padding: 15px;
            border-radius: 12px;
            background-color: #1f2937;
            color: white;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)


st.markdown('<p class="main-title">🏏 IPL Win Predictor</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Predict the winning probability of the chasing team based on the current match state.</p>',
    unsafe_allow_html=True
)


st.sidebar.header("Match Inputs")

batting_team = st.sidebar.selectbox("Batting Team", teams, index=0)
bowling_team = st.sidebar.selectbox("Balling Team", teams, index=1 if len(teams) > 1 else 0)
city = st.sidebar.selectbox("City", cities)

target = st.sidebar.number_input("Target", min_value=1, max_value=300, value=180, step=1)
score = st.sidebar.number_input("Current Score", min_value=0, max_value=300, value=120, step=1)
overs_completed = st.sidebar.number_input(
    "Overs Completed (e.g. 14.2)",
    min_value=0.0,
    max_value=19.5,
    value=14.2,
    step=0.1,
    format="%.1f"
)
wickets_out = st.sidebar.number_input("Wickets Out", min_value=0, max_value=9, value=4, step=1)

predict_btn = st.sidebar.button("Predict Win Probability")


left_col, right_col = st.columns([1.2, 1])


with left_col:
    st.subheader("Match Setup")
    st.markdown(f"""
    <div class="card">
        <p><b>Batting Team:</b> {batting_team}</p>
        <p><b>Bowling Team:</b> {bowling_team}</p>
        <p><b>City:</b> {city}</p>
        <p><b>Target:</b> {target}</p>
        <p><b>Current Score:</b> {score}</p>
        <p><b>Overs Completed:</b> {overs_completed}</p>
        <p><b>Wickets Out:</b> {wickets_out}</p>
    </div>
    """, unsafe_allow_html=True)


with right_col:
    st.subheader("Prediction Output")

    if predict_btn:
        try:
            if batting_team == bowling_team:
                st.error("Batting team and bowling team cannot be the same.")
            else:
                result = predict_win_probability(
                    batting_team=batting_team,
                    bowling_team=bowling_team,
                    city=city,
                    target=target,
                    score=score,
                    overs_completed=overs_completed,
                    wickets_out=wickets_out
                )

                batting_prob = result["batting_team_win_probability"]
                bowling_prob = result["bowling_team_win_probability"]
                input_df = result["input_df"]

                runs_left = int(input_df["runs_left"].iloc[0])
                balls_left = int(input_df["balls_left"].iloc[0])
                wickets_left = int(input_df["wickets_left"].iloc[0])
                crr = float(input_df["crr"].iloc[0])
                rrr = float(input_df["rrr"].iloc[0])

                # Prediction cards
                st.markdown(f"""
                <div class="card">
                    <div class="metric-title">{batting_team} Win Probability</div>
                    <div class="metric-value">{batting_prob}%</div>
                </div>
                """, unsafe_allow_html=True)

                st.progress(min(int(batting_prob), 100))

                st.markdown(f"""
                <div class="card">
                    <div class="metric-title">{bowling_team} Win Probability</div>
                    <div class="metric-value">{bowling_prob}%</div>
                </div>
                """, unsafe_allow_html=True)

                st.progress(min(int(bowling_prob), 100))

                st.subheader("Derived Match State")

                c1, c2, c3, c4, c5 = st.columns(5)
                with c1:
                    st.markdown(f"""
                    <div class="small-box">
                        <div>Runs Left</div>
                        <h3>{runs_left}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
                    <div class="small-box">
                        <div>Balls Left</div>
                        <h3>{balls_left}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                with c3:
                    st.markdown(f"""
                    <div class="small-box">
                        <div>Wickets Left</div>
                        <h3>{wickets_left}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                with c4:
                    st.markdown(f"""
                    <div class="small-box">
                        <div>CRR</div>
                        <h3>{crr:.2f}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                with c5:
                    st.markdown(f"""
                    <div class="small-box">
                        <div>RRR</div>
                        <h3>{rrr:.2f}</h3>
                    </div>
                    """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.info("Enter match details in the sidebar and click **Predict Win Probability**.")