"""
This script contains 8 function definitions:
1. load_data
2. plot_demand
3. prep_df
4. clean_data
5. split_data
6. get_predictions
7. show_predictions
8. demandkBtu_model

which used together build a baseline model for Electricity_Demand_kBtu,
ChilledWater_Demand_kBtu, and Steam_Demand_kBtu.

The functions depend on the following files:
1. elec_tags.csv
2. chw_tags.csv
3. steam_tags.csv
4. wifi_matched_tags.csv 
"""
###############################################################################
# load_data: returns demand_kBtu data frame with indicated data specifications
# option to indicate a different year other than the predefined as optimal
def load_data(tags, start = 'NA', end = 'NA'):

    # determining the utility we are under
    chw = [string for string in tags if string.endswith("ChilledWater_Demand_kBtu")]
    steam = [string for string in tags if string.endswith("Steam_Demand_kBtu")]
    elec = [string for string in tags if string.endswith("Electricity_Demand_kBtu")]

    # load predefined data frames containing year information on which to train the model
    if (start == 'NA') | (end == 'NA'):
        if elec:
            elec_tags = pd.read_csv('elec_tags.csv')
            start = elec_tags[elec_tags['Electricity_Demand_kBtu'] == tags[0]]['start'].tolist()[0]
            end = elec_tags[elec_tags['Electricity_Demand_kBtu'] == tags[0]]['end'].tolist()[0]

        if chw:
            chw_tags = pd.read_csv('chw_tags.csv')
            start = chw_tags[chw_tags['ChilledWater_Demand_kBtu'] == tags[0]]['start'].tolist()[0]
            end = chw_tags[chw_tags['ChilledWater_Demand_kBtu'] == tags[0]]['end'].tolist()[0]

        if steam:
            steam_tags = pd.read_csv('steam_tags.csv')
            start = steam_tags[steam_tags['Steam_Demand_kBtu'] == tags[0]]['start'].tolist()[0]
            end = steam_tags[steam_tags['Steam_Demand_kBtu'] == tags[0]]['end'].tolist()[0]

    # load year data from pi client
    calc = 'summary'
    interval = '1h'
    chunk_size = 10
    weight = 'TimeWeighted'
    summary_calc = 'average'
    max_count = round(1500000/len(tags))

    df = pi.get_stream_by_point(tags, start=start, end=end, _convert_cols=False, calculation=calc,
                                interval=interval, _weight=weight, _summary_type=summary_calc, _max_count=max_count,
                                _chunk_size=chunk_size)
    return df


###############################################################################
# plot_demand: Plots demand_kBtu values for exploratory analysis df must be
# defined previous to function call
def plot_demand(tags, title = 'Utility Demand kBtu'):

    import matplotlib as mpl
    mpl.style.use('seaborn-whitegrid')

    tag = [string for string in tags if string.endswith("_Demand_kBtu")]

    plt.figure(figsize = (12,8))
    plt.plot(df[tag], color = '#F18F01', linewidth = 0.5, alpha = 0.7)
    plt.title(title, fontsize = 16)
    plt.xlabel('Dates', fontsize = 12)
    plt.ylabel('Demand kBtu', fontsize = 12)


###############################################################################
# prep_df: collects and transforms data as needed for each utility to build
# predictive model. Returns data frame ready to be cleaned
def prep_df(df):
    # determining the utility we are under
    chw = [string for string in df.columns if string.endswith("ChilledWater_Demand_kBtu")]
    steam = [string for string in df.columns if string.endswith("Steam_Demand_kBtu")]
    elec = [string for string in df.columns if string.endswith("Electricity_Demand_kBtu")]

    # add outside temperature regardless of the utility
    tag = pi.search_by_point(['aiTIT4045'])
    start = df.index[0]
    end = df.index[-1]
    calc = 'summary'
    interval = '1h'
    chunk_size = 10
    weight = 'TimeWeighted'
    summary_calc = 'average'
    max_count = round(1500000/len(tag))

    temp = pi.get_stream_by_point(tag, start=start, end=end, _convert_cols=False, calculation=calc,
                                interval=interval, _weight=weight, _summary_type=summary_calc, _max_count=max_count,
                                _chunk_size=chunk_size)

    df['aiTIT4045'] = temp.aiTIT4045

    # add holidays variable regardless of utility
    tag = pi.search_by_point(['ACE_Holidays_Boolean'])
    start = df.index[0]
    end = df.index[-1]
    calc = 'summary'
    interval = '1h'
    chunk_size = 10
    weight = 'TimeWeighted'
    summary_calc = 'average'
    max_count = round(1500000/len(tag))

    hol = pi.get_stream_by_point(tag, start=start, end=end, _convert_cols=False, calculation=calc,
                                interval=interval, _weight=weight, _summary_type=summary_calc, _max_count=max_count,
                                _chunk_size=chunk_size)

    df['Holiday'] = hol.ACE_Holidays_Boolean

    #add hour regardless of utility
    df.reset_index(inplace=True)
    df['Hour'] = pd.to_datetime(df['Timestamp']).apply(lambda x: '{hour}'.format(hour = x.hour))
    df['Hour'] = df['Hour'].astype('int64')

    # add Heating Degree Hour, Cooling Degree Hour suqared and month if this
    # is a chilled water model
    if chw:
        #add HDH
        df['HDH'] = 65 - df['aiTIT4045']
        df.loc[df['HDH'] < 0, 'HDH'] = 0

        #add CDH
        df['CDH'] = (df['aiTIT4045'] - 65)**2
        df.loc[df['CDH'] < 0, 'CDH'] = 0

        #add month
        df['Month'] = pd.to_datetime(df['Timestamp']).apply(lambda x: '{month}'.format(month = x.month))
        df['Month'] = df['Month'].astype('int64')

    # add Heating Degree Hour squared, Cooling Degree Hour, weekend indicator
    # variable and month if this is a steam model
    if steam:
        # add HDH
        df['HDH'] = (65 - df['aiTIT4045'])**2
        df.loc[df['HDH'] < 0, 'HDH'] = 0

        # add CDH
        df['CDH'] = df['aiTIT4045'] - 65
        df.loc[df['CDH'] < 0, 'CDH'] = 0

        # add weekend/weekday
        df['DayOfWeek'] = df['Timestamp'].dt.strftime('%w')
        df['DayOfWeek'] = df['DayOfWeek'].astype('int64')
        df['Weekend'] = 0
        df.loc[(df['DayOfWeek'] == 0) | (df['DayOfWeek'] == 6), 'Weekend'] = 1
        df.loc[(df['DayOfWeek'] == 1) | (df['DayOfWeek'] == 2) | (df['DayOfWeek'] == 3) | (df['DayOfWeek'] == 4) | (df['DayOfWeek'] == 5), 'Weekend'] = 0
        df.drop(['DayOfWeek'], inplace = True, axis = 1)

        # add month
        df['Month'] = pd.to_datetime(df['Timestamp']).apply(lambda x: '{month}'.format(month = x.month))
        df['Month'] = df['Month'].astype('int64')

    # add Heating Degree Hour, Cooling Degree Hour and if available WiFi
    # occupancy if this is an electricity model
    if elec:
        #add HDH
        df['HDH'] = 65 - df['aiTIT4045']
        df.loc[df['HDH'] < 0, 'HDH'] = 0

        #add CDH
        df['CDH'] = df['aiTIT4045'] - 65
        df.loc[df['CDH'] < 0, 'CDH'] = 0

        # load predefined data frame to find if building has WiFi occupancy data
        elec_tags = pd.read_csv('elec_tags.csv')
        row = elec_tags.loc[elec_tags['Electricity_Demand_kBtu'] == elec[0]]
        status = row.Status.tolist()

        if status == 'complete':
            # add WiFi
            # load predefined data frame containing matched WiFi tags since names are inconsistent
            wifi_matched_tags = pd.read_csv('wifi_matched_tags.csv')

            wifi_tag = wifi_matched_tags[wifi_matched_tags['Electricity_Tag'] == elec[0]]['Wi-Fi Tags'].to_list()

            df.set_index('Timestamp',inplace=True)
            start = df.index[0]
            end = df.index[-1]
            calc = 'summary'
            interval = '1h'
            chunk_size = 10
            weight = 'TimeWeighted'
            summary_calc = 'average'
            max_count = round(1500000/len(wifi_tag))

            wifi = pi.get_stream_by_point(wifi_tag, start=start, end=end, _convert_cols=False, calculation=calc,
                                        interval=interval, _weight=weight, _summary_type=summary_calc, _max_count=max_count,
                                        _chunk_size=chunk_size)

            # Certain buildings contain multiple WiFi occupancy locations. Add those that do
            if len(wifi_tag) > 1:
                col = np.zeros(len(wifi))
                for i in wifi_tag:
                    col = wifi[i] + col
                df['Wifi_Occupancy_Count'] = col
            else:
                df['Wifi_Occupancy_Count'] = wifi[[wifi_tag[0]]]

            df.reset_index(inplace = True)

    return df


###############################################################################
# clean_data: performs data cleaning according to the needs of each utility
def clean_data(df):
    demand = [string for string in df.columns if string.endswith("_Demand_kBtu")]
    # generaly remove negative and zero values as well as temperatures that do not make sense
    df = df[df[demand[0]] > 0]
    df = df[(df['aiTIT4045'] > 0) & (df['aiTIT4045'] < 120)]
    # note only steam and electricity have a special condition to check
    steam = [string for string in df.columns if string.endswith("Steam_Demand_kBtu")]
    elec = [string for string in df.columns if string.endswith("Electricity_Demand_kBtu")]

    if steam:
        # remove spikes in steam that go over 6500
        df = df[df[steam[0]] < 6500]

    df.dropna(axis=0, inplace=True)
    df.set_index('Timestamp',inplace = True)

    if elec:
        elec_tags = pd.read_csv('elec_tags.csv')
        row = elec_tags.loc[elec_tags['Electricity_Demand_kBtu'] == elec[0]]
        status = row.Status.tolist()

        # continue depending on the presence of WiFi variable
        if status == 'complete':
            wifi = [string for string in df.columns if string.endswith("_Count")]
            df_out = df[[demand[0], 'aiTIT4045', wifi[0]]].copy()
        else:
            df_out = df[[demand[0], 'aiTIT4045']].copy()

    else:
        df_out = df[[demand[0], 'aiTIT4045']].copy()

    # remove outliers
    IQR = df_out.quantile(0.75) - df_out.quantile(0.25)
    df_final = df_out[~((df_out < (df_out.quantile(0.25) - 3 * IQR)) | (df_out > (df_out.quantile(0.75) + 3 * IQR)))]

    df_final.dropna(axis=0, inplace=True)

    df_final.reset_index(inplace=True)
    df = df.loc[df_final['Timestamp']]
    df.reset_index(inplace = True)

    df.drop(['aiTIT4045'], inplace=True, axis=1)

    return df


###############################################################################
# split_data: use sklearn train_test_split to split data
def split_data(df):
    from sklearn.model_selection import train_test_split
    demand = [string for string in df.columns if string.endswith("_Demand_kBtu")]

    x = df.drop(demand, axis = 1)
    y = df[demand].values.reshape(-1,1)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, shuffle = False)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size = 0.5, shuffle = False)

    # mantain dates for later use
    dates_train = X_train['Timestamp']
    dates_test = X_test['Timestamp']
    dates_val = X_val['Timestamp']
    X_train.drop('Timestamp', axis = 1, inplace = True)
    X_test.drop('Timestamp', axis = 1, inplace = True)
    X_val.drop('Timestamp', axis = 1, inplace = True)

    return X_train, X_test, X_val, y_train, y_test, y_val, dates_train, dates_test, dates_val



###############################################################################
# get_predictions: builds and fits an optimized model on one year of data and
# predicts. Results are measured and there are two cases for the predictions.
# If test = True it predicts on the last month of the year selected. If
# test = false it calculates results for training, test and validation sets
def get_predictions(X_train = 'NA', X_test = 'NA', X_val = 'NA', y_train = 'NA', y_test = 'NA', y_val = 'NA', test = False, last_month_x = 'NA', last_month_y = 'NA'):
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import RandomizedSearchCV

    # parameters to use in tuning
    param_grid = {'criterion':['friedman_mse', 'mse'], 'loss':['ls','lad'],'learning_rate':[0.1,0.01, 0.001, 0.05],
                  'n_estimators':np.linspace(10, 200).astype(int), 'random_state':[1], 'min_samples_split': [2, 5, 10],
                 'max_depth': [None] + list(np.linspace(3, 20).astype(int)), 'max_leaf_nodes': [None] + list(np.linspace(10, 50, 500).astype(int))}

    estimator = GradientBoostingRegressor()

    # randomized search to optimized model
    rs = RandomizedSearchCV(estimator, param_grid, n_jobs = -1,
                            scoring = 'neg_root_mean_squared_error', cv = 5,
                            n_iter = 10, verbose = 1)
    rs.fit(X_train, y_train)

    # save best parameters
    parameters = rs.best_params_

    # build and fit optimized model
    model = GradientBoostingRegressor(**parameters)
    model.fit(X_train, y_train)

    from sklearn import metrics
    from statistics import mean

    if test == True:
        #predict on last month of data
        pred_month = model.predict(last_month_x)
        is_ = list()

        # remove negative predictions since it's not meaningful. Added to improve
        # the results of those steam and chilled water models that did not pass the threshhold
        for i in range(len(pred_month)):
            if pred_month[i] > 0:
                is_.append(i)

        last_month_y1 = list()
        pred_month1 = list()
        for i in is_:
            last_month_y1.append(last_month_y[i])
            pred_month1.append(pred_month[i])
        last_month_y1 = np.array(last_month_y1)

        # calculate result metrics
        mse_month = metrics.mean_squared_error(last_month_y1, pred_month1)
        rmse_month = np.sqrt(metrics.mean_squared_error(last_month_y1, pred_month1))
        mae_month = metrics.mean_absolute_error(last_month_y1, pred_month1)
        cv_rmse_month = rmse_month / mean(last_month_y1.flatten())
        r2_month = metrics.r2_score(last_month_y1, pred_month1)

        results = pd.DataFrame(np.array([['MSE', mse_month], ['RMSE', rmse_month], ['MAE', mae_month], ['CV(RMSE)', cv_rmse_month], ['R-squared', r2_month]]), columns=['Measure', 'Last_Month'])

        return pred_month, results

    else:

        # calculate predicted values for training, test, and validation sets
        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)
        pred_val = model.predict(X_val)

        # calculate metrics fro training, test, and validations sets
        #train
        mse_train = metrics.mean_squared_error(y_train1, pred_train1)
        rmse_train = np.sqrt(metrics.mean_squared_error(y_train1, pred_train1))
        mae_train = metrics.mean_absolute_error(y_train1, pred_train1)
        cv_rmse_train = rmse_train / mean(y_train1.flatten())
        r2_train = metrics.r2_score(y_train1, pred_train1)

        #test
        mse_test = metrics.mean_squared_error(y_test, pred_test)
        rmse_test = np.sqrt(metrics.mean_squared_error(y_test, pred_test))
        mae_test = metrics.mean_absolute_error(y_test, pred_test)
        cv_rmse_test = rmse_test / mean(y_test.flatten())
        r2_test = metrics.r2_score(y_test, pred_test)

        #val
        mse_val = metrics.mean_squared_error(y_val, pred_val)
        rmse_val = np.sqrt(metrics.mean_squared_error(y_val, pred_val))
        mae_val = metrics.mean_absolute_error(y_val, pred_val)
        cv_rmse_val = rmse_val / mean(y_val.flatten())
        r2_val = metrics.r2_score(y_val, pred_val)

        results = pd.DataFrame(np.array([['MSE', mse_train, mse_test, mse_val], ['RMSE', rmse_train, rmse_test, rmse_val], ['MAE', mae_train, mae_test, mae_val], ['CV(RMSE)', cv_rmse_train, cv_rmse_test, cv_rmse_val], ['R-squared', r2_train, r2_test, r2_val]]), columns=['Measure', 'Train', 'Test', 'Val'])

        return pred_train, pred_test, pred_val, results


###############################################################################
# show_predictions: visualize predicted against true values along with the most
# important metrics
def show_predictions(y_train = 'NA', y_test = 'NA', y_val = 'NA', pred_train = 'NA', pred_test = 'NA', pred_val = 'NA', dates_train = 'NA', dates_test = 'NA', dates_val = 'NA', test = False, last_month_y = 'NA', pred_month = 'NA', dates_month = 'NA', tag = ''):

    from sklearn import metrics
    from statistics import mean

    import matplotlib as mpl
    mpl.style.use('seaborn-whitegrid')

    # if test = True visualize only the last month of data predictions and actual values
    if test == True:
        import numpy as np
        is_ = list()

        for i in range(len(pred_month)):
            if pred_month[i] > 0:
                is_.append(i)

        last_month_y1 = list()
        pred_month1 = list()
        for i in is_:
            last_month_y1.append(last_month_y[i])
            pred_month1.append(pred_month[i])
        last_month_y1 = np.array(last_month_y1)

        # calculate metrics to use in plot
        mse_month = metrics.mean_squared_error(last_month_y1, pred_month1)
        rmse_month = np.sqrt(metrics.mean_squared_error(last_month_y1, pred_month1))
        mae_month = metrics.mean_absolute_error(last_month_y1, pred_month1)
        cv_rmse_month = rmse_month / mean(last_month_y1.flatten())
        r2_month = metrics.r2_score(last_month_y1, pred_month1)


        plt.figure(figsize = (12, 8))
        plt.plot(dates_month, last_month_y, color='#F18F01', linewidth=0.5, alpha = 0.7)
        plt.plot(dates_month, pred_month, color='#006E90', linewidth=0.5, alpha = 0.9)

        plt.title(f'GradientBoostingRegressor: Predicted vs True Values \n Last Month of {tag} Data \n MSE: {round(mse_month, 4)}     MAE: {round(mae_month,4)}    RMSE: {round(rmse_month, 4)}  \n CV(RMSE): {round(cv_rmse_month, 4)}     R2:{round(r2_month,4)}', fontsize = 16)
        plt.xlabel('Dates', fontsize = 12)
        plt.ylabel('Demand kBtu', fontsize = 12)

        leg = plt.legend(labels = ['True','Predicted'], loc = "upper right", bbox_to_anchor=(1.2, 1), fontsize = 12)

        for line in leg.get_lines():
            line.set_linewidth(5.0)

    # if test = False, visualize training, test and validation sets predicted against true values
    else:
        # calculate metrics to use in plot
        #train
        mse_train = metrics.mean_squared_error(y_train, pred_train)
        rmse_train = np.sqrt(metrics.mean_squared_error(y_train, pred_train))
        mae_train = metrics.mean_absolute_error(y_train, pred_train)
        cv_rmse_train = rmse_train / mean(y_train.flatten())
        r2_train = metrics.r2_score(y_train, pred_train)

        #test
        mse_test = metrics.mean_squared_error(y_test, pred_test)
        rmse_test = np.sqrt(metrics.mean_squared_error(y_test, pred_test))
        mae_test = metrics.mean_absolute_error(y_test, pred_test)
        cv_rmse_test = rmse_test / mean(y_test.flatten())
        r2_test = metrics.r2_score(y_test, pred_test)

        #val
        mse_val = metrics.mean_squared_error(y_val, pred_val)
        rmse_val = np.sqrt(metrics.mean_squared_error(y_val, pred_val))
        mae_val = metrics.mean_absolute_error(y_val, pred_val)
        cv_rmse_val = rmse_val / mean(y_val.flatten())
        r2_val = metrics.r2_score(y_val, pred_val)


        plt.figure(figsize = (12, 30))

        plt.subplot(3,1,1)
        plt.plot(dates_train, y_train, color='#F18F01', linewidth=0.5, alpha = 0.7)
        plt.plot(dates_train, pred_train, color='#006E90', linewidth=0.5, alpha = 0.9)

        plt.title(f'GradientBoostingRegressor: Predicted vs True Values \n Training Data \n MSE: {round(mse_train, 4)}     MAE: {round(mae_train,4)}    RMSE: {round(rmse_train, 4)}  \n CV(RMSE): {round(cv_rmse_train, 4)}     R2:{round(r2_train,4)}', fontsize = 16)
        plt.xlabel('Dates', fontsize = 12)
        plt.ylabel('Demand kBtu', fontsize = 12)

        leg = plt.legend(labels = ['True','Predicted'], loc = "upper right", bbox_to_anchor=(1.2, 1), fontsize = 12)

        for line in leg.get_lines():
            line.set_linewidth(5.0)

        plt.subplot(3,1,2)
        plt.plot(dates_test, y_test, color='#F18F01', linewidth=0.5, alpha = 0.7)
        plt.plot(dates_test, pred_test, color='#006E90', linewidth=0.5, alpha = 0.9)

        plt.title(f'Testing Data \n MSE: {round(mse_test, 4)}     MAE: {round(mae_test,4)}    RMSE: {round(rmse_test, 4)} \n CV(RMSE): {round(cv_rmse_test, 4)}    R2:{round(r2_test,4)}', fontsize = 16)
        plt.xlabel('Dates', fontsize = 12)
        plt.ylabel('Demand kBtu', fontsize = 12)

        leg = plt.legend(labels = ['True','Predicted'], loc = "upper right", bbox_to_anchor=(1.2, 1), fontsize = 12)

        for line in leg.get_lines():
            line.set_linewidth(5.0)

        plt.subplot(3,1,3)
        plt.plot(dates_val, y_val, color='#F18F01', linewidth=0.5, alpha = 0.7)
        plt.plot(dates_val, pred_val, color='#006E90', linewidth=0.5, alpha = 0.9)

        plt.title(f'Validation Data \n MSE: {round(mse_val, 4)}     MAE: {round(mae_val,4)}     RMSE: {round(rmse_val, 4)} \n CV(RMSE): {round(cv_rmse_val, 4)}     R2:{round(r2_val,4)}', fontsize = 16)
        plt.xlabel('Dates', fontsize = 12)
        plt.ylabel('Demand kBtu', fontsize = 12)


        leg = plt.legend(labels = ['True','Predicted'], loc = "upper right", bbox_to_anchor=(1.2, 1), fontsize = 12)

        for line in leg.get_lines():
            line.set_linewidth(5.0)



###############################################################################
# demandkBtu_model: puts together previous functions to implement the process
# from beginning to end. returns visualizations and results depending of the value for 'test'
def demandkBtu_model(tags, test = False, start = 'NA', end = 'NA'):
    import pandas as pd
    # loads predefined data frames containing tag information such as year of
    # optimal data and quality of model/data
    elec_tags = pd.read_csv('elec_tags.csv')
    chw_tags = pd.read_csv('chw_tags.csv')
    steam_tags = pd.read_csv('steam_tags.csv')
    elec_tags = elec_tags[elec_tags['Status'] != 'no data']
    chw_tags = chw_tags[chw_tags['Status'] != 'no data']
    steam_tags = steam_tags[steam_tags['Status'] != 'no data']
    possible_tags = elec_tags['Electricity_Demand_kBtu'].tolist() + chw_tags['ChilledWater_Demand_kBtu'].tolist() + steam_tags['Steam_Demand_kBtu'].tolist()

    # confirm the tag exists
    if len(pi.search_by_point(tags[0])) == 0:
        return print('Tag does not exist')

    # if status = bad data the corresponding tags contain data of low quality to attempt the process
    if tags[0] not in possible_tags:
        return print(tags[0], 'contains unreliable data or no data at all')

    # if status = bad model the corresponding tags have been through the process and returned bad/unreliable models
    if (chw_tags.loc[chw_tags['ChilledWater_Demand_kBtu'] == tags[0]]['Status'].tolist() == 'bad model') | (elec_tags.loc[elec_tags['Electricity_Demand_kBtu'] == tags[0]]['Status'].tolist() == 'bad model') | (steam_tags.loc[steam_tags['Steam_Demand_kBtu'] == tags[0]]['Status'].tolist() == 'bad model'):
        return print(tags[0], 'produces an unreliable model with CV(RMSE) over 0.35 or R-squared under 0.75')

    # obtained the data frame to implement in model
    df = clean_data(prep_df(load_data(tags, start, end)))

    # if test = True we predict the last month of observations only from the year of optimal data
    if test == True:
        # separate last month of observations
        last = len(df) - 1
        first = last - 730
        last_month = df.iloc[np.r_[first:last]]

        # split variables to use as testing data set in get_predictions
        last_month_x = last_month.drop(tags[0], axis = 1)
        last_month_y = last_month[tags[0]].values.reshape(-1,1)
        dates_month = last_month['Timestamp']
        dates_month.reset_index(inplace = True, drop = True)
        last_month_x.drop('Timestamp', axis = 1, inplace = True)

        # only interested in X_train and y_train
        X_train, X_test, X_val, y_train, y_test, y_val, dates_train, dates_test, dates_val = split_data(df)

        # build model using x_train y_train for training and last_month_x, last_month_y for testing
        pred_month, results = get_predictions(X_train = X_train, y_train = y_train, test = True, last_month_y = last_month_y, last_month_x = last_month_x)

        #show_predictions(last_month_y = last_month_y, pred_month = pred_month, dates_month = dates_month, test = True, tag = tags)

        # returns predictions, metrics and dates of said predictions
        return pred_month, results.Last_Month.tolist(), dates_month


    else:
        # splits the data
        X_train, X_test, X_val, y_train, y_test, y_val, dates_train, dates_test, dates_val = split_data(df)
        # builds, trains, test the model
        pred_train, pred_test, pred_val, results = get_predictions(X_train, X_test, X_val, y_train, y_test, y_val)
        # visualized the results
        show_predictions(y_train, y_test, y_val, pred_train, pred_test, pred_val, dates_train, dates_test, dates_val)



""" To build a model run: demandkBtu_model(tag, test = True)
It will return 3 lists: predictions, results, dates
If test = False it returns visualizations of predictions and display model
metrics for training, testing and validation sets
"""
