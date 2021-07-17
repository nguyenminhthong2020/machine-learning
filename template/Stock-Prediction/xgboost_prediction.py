
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

# Mute sklearn warnings
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=DeprecationWarning)

def xgboost_prediction():
    #Read data
    df=pd.read_csv("./assets/NSE-TATA.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[(df['Date'].dt.year >= 2010)].copy()
    df.index = range(len(df))

    # Get month of each sample
    df['Month'] = df['Date'].dt.month

    # Sort by datetime
    df.sort_values(by='Date', inplace=True, ascending=True)

    df['Close'] = df['Close'].shift(-1)

    df = df.iloc[33:] # Because of moving averages and MACD line
    df = df[:-1]      # Because of shifting close price

    df.index = range(len(df))

    # Split data
    test_size  = 0.1

    test_split_idx  = int(df.shape[0] * (1-test_size))

    train_df  = df.loc[:test_split_idx].copy()
    test_df   = df.loc[test_split_idx+1:].copy()

    # Drop some columns
    drop_cols = ['Date', 'Volume', 'Open', 'Low', 'High']

    train_df = train_df.drop(drop_cols, 1)
    test_df  = test_df.drop(drop_cols, 1)

    y_train = train_df['Close'].copy()
    X_train = train_df.drop(['Close'], 1)

    y_test  = test_df['Close'].copy()
    X_test  = test_df.drop(['Close'], 1)

    parameters = {
        'n_estimators': [100, 200, 300, 400],
        'learning_rate': [0.001, 0.005, 0.01, 0.05],
        'max_depth': [8, 10, 12, 15],
        'gamma': [0.001, 0.005, 0.01, 0.02],
        'random_state': [42]
    }

    eval_set = [(X_train, y_train)]
    model = xgb.XGBRegressor(eval_set=eval_set, objective='reg:squarederror', verbose=False)
    clf = GridSearchCV(model, parameters)

    clf.fit(X_train, y_train)

    model = xgb.XGBRegressor(**clf.best_params_, objective='reg:squarederror')
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    y_pred = model.predict(X_test)

    xgb_train=df.loc[:test_split_idx].copy()
    xgb_valid = df.loc[test_split_idx+1:].copy()
    xgb_valid['Close'] = y_pred

    return xgb_train, xgb_valid