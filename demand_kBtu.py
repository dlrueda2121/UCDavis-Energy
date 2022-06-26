
#### load_data: returns demand_kBtu data frame with indicated data specifications
def load_data(tags, chw, steam, elec, start = 'NA', end = 'NA'):


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


###### prep_df: collects and transforms data as needed for each utility to build predictive model. Returns data frame ready to be cleaned
def prep_df(df, chw, steam, elec, train = True):
    if train:
        # determine start and end date for training data:
        start = df.index[0]
        end = df.index[-1]
    else:
        start = '*-1mo'
        end = '*'

    # add outside temperature regardless of the utility
    tag = pi.search_by_point(['aiTIT4045'])
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

    # add Heating Degree Hour, Cooling Degree Hour suqared and month if this is a chilled water model
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

    # add Heating Degree Hour squared, Cooling Degree Hour, weekend indicator variable and month if this is a steam model
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

    # add Heating Degree Hour, Cooling Degree Hour and if available WiFi occupancy if this is an electricity model
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

        if status[0] == 'complete':
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

            wifi.reset_index(inplace = True)
            df.reset_index(inplace = True)
            #df = df.iloc[:-1 , :]

            # Certain buildings contain multiple WiFi occupancy locations. Add those that do
            if len(wifi_tag) > 1:
                col = np.zeros(len(wifi))
                for i in wifi_tag:
                    col = wifi[i] + col
                df['Wifi_Occupancy_Count'] = col
            else:
                df['Wifi_Occupancy_Count'] = wifi[[wifi_tag[0]]]


            #df.reset_index(inplace = True)


    return df


###### clean_data: performs data cleaning according to the needs of each utility
def clean_data(df, chw, steam, elec, train = True):
    df = df[(df['aiTIT4045'] > 0) & (df['aiTIT4045'] < 120)]
    df.dropna(axis=0, inplace=True)
    df.set_index('Timestamp',inplace = True)


    if train:
        demand_tag = [string for string in df.columns if string.endswith("_Demand_kBtu")]
        # generaly remove negative and zero values as well as temperatures that do not make sense
        df = df[df[demand_tag[0]] > 0]

        if chw:
            df_out = df[[demand_tag[0], 'aiTIT4045']].copy()

        if steam:
            # remove spikes in steam that go over 6500
            df = df[df[steam[0]] < 6500]
            df_out = df[[demand_tag[0], 'aiTIT4045']].copy()

        if elec:
            elec_tags = pd.read_csv('elec_tags.csv')
            row = elec_tags.loc[elec_tags['Electricity_Demand_kBtu'] == elec[0]]
            status = row.Status.tolist()
            if status[0] == 'complete':
                wifi = [string for string in df.columns if string.endswith("_Count")]
                df_out = df[[demand_tag[0], 'aiTIT4045', wifi[0]]].copy()
            else:
                df_out = df[[demand_tag[0], 'aiTIT4045']].copy()

    else:
        df_out = df[['aiTIT4045']].copy()

    # remove outliers
    IQR = df_out.quantile(0.75) - df_out.quantile(0.25)
    df_final = df_out[~((df_out < (df_out.quantile(0.25) - 3 * IQR)) | (df_out > (df_out.quantile(0.75) + 3 * IQR)))]

    df_final.dropna(axis=0, inplace=True)

    df_final.reset_index(inplace=True)
    df = df.loc[df_final['Timestamp']]
    df.reset_index(inplace = True)

    df.drop(['aiTIT4045'], inplace=True, axis=1)

    return df



###### get_predictions: builds and fits an optimized model on one year of data and predicts. Results are measured and there are two cases for the predictions. If test = True it predicts on the last month of the year selected. If test = false it calculates results for training, test and validation sets
def get_predictions(X_train = 'NA', y_train = 'NA', last_month_x = 'NA'):
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

    #predict on last month of data
    pred_month = model.predict(last_month_x)

    # remove negative predictions since it's not meaningful. Added to improve the results of those steam and chilled water models that did not pass the threshhold
    pred_month1 = list()
    for i in range(len(pred_month)):
        if pred_month[i] > 0:
            #is_.append(i)
            pred_month1.append(pred_month[i])
        else:
            pred_month1.append(0)

    return pred_month1


###### demandkBtu_model: puts together previous functions to implement the process from beginning to end. returns visualizations and results depending of the value for 'test'
def demandkBtu_model(tags, start = 'NA', end = 'NA'):
    import pandas as pd
    chw = [string for string in tags if string.endswith("ChilledWater_Demand_kBtu")]
    steam = [string for string in tags if string.endswith("Steam_Demand_kBtu")]
    elec = [string for string in tags if string.endswith("Electricity_Demand_kBtu")]


    # obtain the data frame to train the  model
    df_train = clean_data(prep_df(load_data(tags, chw = chw, steam = steam, elec = elec), train = True, chw = chw, steam = steam, elec = elec), chw = chw, steam = steam, elec = elec)

    # obtain data frame with last month's data to predict demand_kBtu
    last_month_x = pd.DataFrame()
    last_month_x = prep_df(last_month_x, train = False, chw = chw, steam = steam, elec = elec)
    dates_month = last_month_x['Timestamp']
    dates_month.reset_index(inplace = True, drop = True)
    last_month_x = clean_data(last_month_x, train = False, chw = chw, steam = steam, elec = elec)


    # predict the last month of observations
    dates_month2 = last_month_x['Timestamp']
    dates_month2.reset_index(inplace = True, drop = True)

    # split x and y training data
    from sklearn.model_selection import train_test_split

    x = df_train.drop(tags[0], axis = 1)
    y = df_train[tags[0]].values.reshape(-1,1)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.0001, shuffle = False)

    y_dummy = dates_month2.values.reshape(-1,1)
    last_month_x, dummyyy, dummyy, dummy = train_test_split(last_month_x, y_dummy, test_size = 0.001, shuffle = False)

    dates_month2 = last_month_x['Timestamp']
    dates_month2.reset_index(inplace = True, drop = True)
    last_month_x.drop('Timestamp', axis = 1, inplace = True)

    X_train.drop('Timestamp', axis = 1, inplace = True)

    # build model using data for training and last_month_x for predictions
    pred_month = get_predictions(X_train = X_train, y_train = y_train, last_month_x = last_month_x)

    # returns predictions, dates of said predictions
    return pred_month, dates_month


################################
elec_tags = pd.read_csv('elec_tags.csv')
chw_tags = pd.read_csv('chw_tags.csv')
steam_tags = pd.read_csv('steam_tags.csv')
elec_tags = elec_tags[(elec_tags['Status'] == 'complete') | (elec_tags['Status'] == 'incomplete')]
chw_tags = chw_tags[chw_tags['Status'] == 'complete']
steam_tags = steam_tags[steam_tags['Status'] == 'complete']
possible_tags = elec_tags['Electricity_Demand_kBtu'].tolist() + chw_tags['ChilledWater_Demand_kBtu'].tolist() + steam_tags['Steam_Demand_kBtu'].tolist()

#dataframe for predictions
predictions_all = pd.DataFrame(index = range(718))

# for status == complete or incomplete
for i in possible_tags:
    tag = list()
    tag.append(i)
    dates = pd.DataFrame()
    predictions = pd.DataFrame()
    predictions[i], dates[i] = demandkBtu_model(tag)
    predictions_all[i] = predictions[[i]]

predictions_all['Timestamp'] = dates[[i]]
predictions_all.set_index('Timestamp', inplace = True)

predictions_all.to_csv('predictions_all.csv', index = False)
