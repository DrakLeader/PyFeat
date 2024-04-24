base = make_pipeline(
    StackingEstimator(estimator=LassoLarsCV(normalize=True)),
    StackingEstimator(
        estimator=LinearSVR(
            c=0.01, dual=True, epsilon=0.001, loss="epsilon_insensitive", tol=0.1
        )
    ),
    RobustScaler(),
    StackingEstimator(
        estimator=XGBRegressor(
            learning_rate=0.1,
            max_depth=1,
            min_child_weight=6,
            n_estimators=100,
            nthread=l,
            objective="regsquarederror",
            subsample=0.6500000000000001,
        )
    ),
    MinMaxScaler(),
    RandomForestRegressor(
        bootstrap=False,
        max_features=0.05,
        min_samples_leaf=1,
        min_samples_split=4,
        n_estimators=100,
    ),
)
parameters = {
    "base_estimator": base,
    "n_estimators": 50,
    "learning_rate": 0.3,
    "loss": "linear",
    "random_state": 9,
}
model = AdaBoostRegressor(
    base_estimator=parameters["base_estimator"],
    n_estimators=parameters["n_estimators"],
    learning_rate=parameters["learning_rate"],
    loss=parameters["loss"],
    random_state=parameters["random_state"],
)
