import os
import pickle
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    GroupKFold,
    cross_validate,
    RandomizedSearchCV
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, brier_score_loss

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
ARTIFACTS_DIR = "artifacts"
MATCHES_PATH = "data/matches.csv"
DELIVERIES_PATH = "data/deliveries.csv"


def load_data(matches_path=MATCHES_PATH, deliveries_path=DELIVERIES_PATH):
    match = pd.read_csv(matches_path)
    delivery = pd.read_csv(deliveries_path)

    teams = [
        'Royal Challengers Bengaluru',
        'Gujarat Titans',
        'Sunrisers Hyderabad',
        'Rajasthan Royals',
        'Punjab Kings',
        'Delhi Capitals',
        'Kolkata Knight Riders',
        'Chennai Super Kings',
        'Mumbai Indians',
        'Lucknow Super Giants'
    ]

    rename_map = {
        'Royal Challengers Bangalore': 'Royal Challengers Bengaluru',
        'Delhi Daredevils': 'Delhi Capitals',
        'Kings XI Punjab': 'Punjab Kings',
    }

    for col in ['team1', 'team2']:
        match[col] = match[col].replace(rename_map)

    match = match[match['team1'].isin(teams) & match['team2'].isin(teams)].copy()

    for col in ['batting_team', 'bowling_team']:
        delivery[col] = delivery[col].replace(rename_map)

    delivery = delivery[
        delivery['batting_team'].isin(teams) &
        delivery['bowling_team'].isin(teams)
    ].copy()

    return match, delivery, teams


def build_features(match, delivery):
    total_run_df = (
        delivery.groupby(['match_id', 'inning'], as_index=False)['total_runs']
        .sum()
    )
    total_run_df = total_run_df[total_run_df['inning'] == 1].copy()

    match_df = match.merge(
        total_run_df[['match_id', 'total_runs']],
        left_on='id',
        right_on='match_id'
    )

    match_df = match_df[['match_id', 'city', 'winner', 'total_runs']].copy()

    delivery_df = match_df.merge(delivery, on='match_id')
    delivery_df = delivery_df[delivery_df['inning'] == 2].copy()
    delivery_df = delivery_df.sort_values(['match_id', 'over', 'ball']).copy()

    delivery_df.rename(columns={
        'total_runs_x': 'target',
        'total_runs_y': 'runs_scored_on_ball'
    }, inplace=True)

    delivery_df['curr_score'] = (
        delivery_df.groupby('match_id')['runs_scored_on_ball'].cumsum()
    )

    delivery_df['balls_left'] = 120 - (
        delivery_df['over'] * 6 + delivery_df['ball']
    )

    delivery_df['runs_left'] = delivery_df['target'] - delivery_df['curr_score']

    delivery_df['player_dismissed'] = delivery_df['player_dismissed'].fillna('0')
    delivery_df['player_dismissed'] = (
        delivery_df['player_dismissed'] != '0'
    ).astype(int)

    wickets_fallen = delivery_df.groupby('match_id')['player_dismissed'].cumsum()
    delivery_df['wickets_left'] = 10 - wickets_fallen

    balls_bowled = 120 - delivery_df['balls_left']
    delivery_df['crr'] = np.where(
        balls_bowled > 0,
        (delivery_df['curr_score'] * 6) / balls_bowled,
        0
    )

    delivery_df['rrr'] = np.where(
        delivery_df['balls_left'] > 0,
        (delivery_df['runs_left'] * 6) / delivery_df['balls_left'],
        0
    )

    delivery_df['result'] = (
        delivery_df['batting_team'] == delivery_df['winner']
    ).astype(int)

    final_df = delivery_df[[
        'match_id',
        'batting_team',
        'bowling_team',
        'city',
        'runs_left',
        'balls_left',
        'wickets_left',
        'target',
        'crr',
        'rrr',
        'result'
    ]].copy()

    final_df.dropna(inplace=True)
    final_df = final_df[final_df['balls_left'] != 0]
    final_df = final_df[final_df['wickets_left'] >= 0]

    return final_df


def split_train_test(final_df):
    match_ids = final_df['match_id'].unique()

    train_ids, test_ids = train_test_split(
        match_ids,
        test_size=0.2,
        random_state=RANDOM_STATE
    )

    train_df = final_df[final_df['match_id'].isin(train_ids)].copy()
    test_df = final_df[final_df['match_id'].isin(test_ids)].copy()

    feature_cols = [
        'batting_team',
        'bowling_team',
        'city',
        'runs_left',
        'balls_left',
        'wickets_left',
        'target',
        'crr',
        'rrr'
    ]

    X_train = train_df[feature_cols]
    y_train = train_df['result']

    X_test = test_df[feature_cols]
    y_test = test_df['result']

    groups_train = train_df['match_id']

    return X_train, X_test, y_train, y_test, groups_train, feature_cols, train_df, test_df


def build_pipelines():
    preprocessor = ColumnTransformer([
        (
            'cat',
            OneHotEncoder(
                handle_unknown='ignore',
                drop='first',
                sparse_output=False
            ),
            ['batting_team', 'bowling_team', 'city']
        )
    ], remainder='passthrough')

    pipe_lr = Pipeline([
        ('prep', preprocessor),
        ('model', LogisticRegression(
            solver='liblinear',
            max_iter=1000
        ))
    ])

    pipe_rf = Pipeline([
        ('prep', preprocessor),
        ('model', RandomForestClassifier(
            random_state=RANDOM_STATE
        ))
    ])

    pipe_gb = Pipeline([
        ('prep', preprocessor),
        ('model', GradientBoostingClassifier(
            random_state=RANDOM_STATE
        ))
    ])

    return pipe_lr, pipe_rf, pipe_gb


def evaluate_with_group_cv(X_train, y_train, groups_train, pipe_lr, pipe_rf, pipe_gb):
    gkf = GroupKFold(n_splits=5)

    scoring = {
        'accuracy': 'accuracy',
        'neg_log_loss': 'neg_log_loss',
        'roc_auc': 'roc_auc'
    }

    models = {
        'Logistic Regression': pipe_lr,
        'Random Forest': pipe_rf,
        'Gradient Boosting': pipe_gb
    }

    print("\n" + "=" * 80)
    print("GROUPED CROSS-VALIDATION RESULTS (TRAIN SET ONLY)")
    print("=" * 80)

    cv_summary = {}

    for name, pipe in models.items():
        scores = cross_validate(
            pipe,
            X_train,
            y_train,
            groups=groups_train,
            cv=gkf,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False
        )

        mean_acc = scores['test_accuracy'].mean()
        std_acc = scores['test_accuracy'].std()
        mean_ll = -scores['test_neg_log_loss'].mean()
        mean_auc = scores['test_roc_auc'].mean()

        cv_summary[name] = {
            'cv_accuracy_mean': mean_acc,
            'cv_accuracy_std': std_acc,
            'cv_log_loss_mean': mean_ll,
            'cv_auc_mean': mean_auc
        }

        print(f"{name}")
        print(f"  CV Accuracy : {mean_acc:.4f} (+/- {std_acc:.4f})")
        print(f"  CV Log Loss : {mean_ll:.4f}")
        print(f"  CV ROC-AUC  : {mean_auc:.4f}")
        print("-" * 80)

    return cv_summary, gkf


def tune_models(X_train, y_train, groups_train, pipe_rf, pipe_gb, gkf):
    print("\n" + "=" * 80)
    print("HYPERPARAMETER TUNING")
    print("=" * 80)

    rf_param_dist = {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [6, 10, 14, None],
        'model__min_samples_leaf': [1, 2, 5, 10]
    }

    gb_param_dist = {
        'model__n_estimators': [100, 200, 300],
        'model__learning_rate': [0.03, 0.05, 0.1],
        'model__max_depth': [2, 3, 4]
    }

    rf_search = RandomizedSearchCV(
        estimator=pipe_rf,
        param_distributions=rf_param_dist,
        n_iter=10,
        cv=gkf,
        scoring='neg_log_loss',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    )

    gb_search = RandomizedSearchCV(
        estimator=pipe_gb,
        param_distributions=gb_param_dist,
        n_iter=10,
        cv=gkf,
        scoring='neg_log_loss',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    )

    print("\nTuning Random Forest...")
    rf_search.fit(X_train, y_train, groups=groups_train)
    print("Best RF Params:", rf_search.best_params_)
    print("Best RF CV Log Loss:", -rf_search.best_score_)

    print("\nTuning Gradient Boosting...")
    gb_search.fit(X_train, y_train, groups=groups_train)
    print("Best GB Params:", gb_search.best_params_)
    print("Best GB CV Log Loss:", -gb_search.best_score_)

    return rf_search.best_estimator_, gb_search.best_estimator_


def evaluate_on_test(X_train, y_train, X_test, y_test, pipe_lr, tuned_rf, tuned_gb):
    print("\n" + "=" * 80)
    print("HELD-OUT TEST SET RESULTS (UNSEEN MATCHES)")
    print("=" * 80)

    candidates = {
        'Logistic Regression': pipe_lr,
        'Random Forest (Tuned)': tuned_rf,
        'Gradient Boosting (Tuned)': tuned_gb
    }

    results = {}
    best_name = None
    best_model = None
    best_log_loss = np.inf

    for name, model in candidates.items():
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        ll = log_loss(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        brier = brier_score_loss(y_test, proba)

        results[name] = {
            'test_accuracy': acc,
            'test_log_loss': ll,
            'test_auc': auc,
            'test_brier': brier
        }

        print(f"{name}")
        print(f"  Test Accuracy : {acc:.4f}")
        print(f"  Test Log Loss : {ll:.4f}")
        print(f"  Test ROC-AUC  : {auc:.4f}")
        print(f"  Test Brier    : {brier:.4f}")
        print("-" * 80)

        if ll < best_log_loss:
            best_log_loss = ll
            best_name = name
            best_model = model

    print(f"\nBest model selected by TEST LOG LOSS: {best_name}")
    print(f"Best Test Log Loss: {best_log_loss:.4f}")

    return best_model, best_name, results


def save_artifacts(best_model, best_model_name, final_df, feature_cols):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    model_path = os.path.join(ARTIFACTS_DIR, "model.pkl")
    metadata_path = os.path.join(ARTIFACTS_DIR, "metadata.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)

    cities = [
        "Ahmedabad",
        "Bangalore",
        "Bengaluru",
        "Chandigarh",
        "Chennai",
        "Delhi",
        "Dharamsala",
        "Hyderabad",
        "Jaipur",
        "Kolkata",
        "Lucknow",
        "Mohali",
        "Mumbai",
        "Navi Mumbai",
        "Pune",
        "Visakhapatnam"
    ]

    metadata = {
        'feature_cols': feature_cols,
        'teams': sorted(final_df['batting_team'].dropna().unique().tolist()),
        'cities': cities,
        'best_model_name': best_model_name
    }

    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)

    print("\n" + "=" * 80)
    print("ARTIFACTS SAVED")
    print("=" * 80)
    print(f"Model saved to    : {model_path}")
    print(f"Metadata saved to : {metadata_path}")


def main():
    print("=" * 80)
    print("IPL WIN PREDICTOR — TRAINING PIPELINE")
    print("=" * 80)

    match, delivery, teams = load_data()
    final_df = build_features(match, delivery)

    print("\nFinal dataset shape:", final_df.shape)
    print("Sample rows:")
    print(final_df.head())

    X_train, X_test, y_train, y_test, groups_train, feature_cols, train_df, test_df = split_train_test(final_df)

    print("\nTrain shape:", X_train.shape)
    print("Test shape :", X_test.shape)
    print("Unique train matches:", train_df['match_id'].nunique())
    print("Unique test matches :", test_df['match_id'].nunique())

    pipe_lr, pipe_rf, pipe_gb = build_pipelines()

    cv_summary, gkf = evaluate_with_group_cv(
        X_train, y_train, groups_train, pipe_lr, pipe_rf, pipe_gb
    )

    tuned_rf, tuned_gb = tune_models(
        X_train, y_train, groups_train, pipe_rf, pipe_gb, gkf
    )

    best_model, best_model_name, test_results = evaluate_on_test(
        X_train, y_train, X_test, y_test, pipe_lr, tuned_rf, tuned_gb
    )

    save_artifacts(best_model, best_model_name, final_df, feature_cols)

    print("\nTraining complete.")


if __name__ == "__main__":
    main()