from django.shortcuts import render
import numpy as np
import pandas as pd


def home(request):
    return render(request, "home.html")


def predict(request):
    return render(request, "predict.html")


def result(request):
    df1 = pd.read_csv(r"C:\Users\rithulaa\Downloads\data.csv")
    df2 = df1.drop(['area_type', 'society', 'balcony', 'availability'], axis='columns')
    df3 = df2.dropna()
    df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))

    def convert_sqft_to_num(x):
        tokens = x.split('-')
        if len(tokens) == 2:
            return (float(tokens[0]) + float(tokens[1])) / 2
        try:
            return float(x)
        except:
            return None

    df4 = df3.copy()
    df4.total_sqft = df4.total_sqft.apply(convert_sqft_to_num)
    df4 = df4[df4.total_sqft.notnull()]
    df5 = df4.copy()
    df5['price_per_sqft'] = df5['price'] * 100000 / df5['total_sqft']
    df5.location = df5.location.apply(lambda x: x.strip())
    location_stats = df5['location'].value_counts(ascending=False)
    location_stats_less_than_10 = location_stats[location_stats <= 10]
    df5.location = df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
    df6 = df5[~(df5.total_sqft / df5.bhk < 300)]

    def remove_pps_outliers(df):
        df_out = pd.DataFrame()
        for key, subdf in df.groupby('location'):
            m = np.mean(subdf.price_per_sqft)
            st = np.std(subdf.price_per_sqft)
            reduced_df = subdf[(subdf.price_per_sqft > (m - st)) & (subdf.price_per_sqft <= (m + st))]
            df_out = pd.concat([df_out, reduced_df], ignore_index=True)
        return df_out

    df7 = remove_pps_outliers(df6)

    def remove_bhk_outliers(df):
        exclude_indices = np.array([])
        for location, location_df in df.groupby('location'):
            bhk_stats = {}
            for bhk, bhk_df in location_df.groupby('bhk'):
                bhk_stats[bhk] = {
                    'mean': np.mean(bhk_df.price_per_sqft),
                    'std': np.std(bhk_df.price_per_sqft),
                    'count': bhk_df.shape[0]
                }
            for bhk, bhk_df in location_df.groupby('bhk'):
                stats = bhk_stats.get(bhk - 1)
                if stats and stats['count'] > 5:
                    exclude_indices = np.append(exclude_indices,
                                                bhk_df[bhk_df.price_per_sqft < (stats['mean'])].index.values)
        return df.drop(exclude_indices, axis='index')

    df8 = remove_bhk_outliers(df7)
    df9 = df8[df8.bath < df8.bhk + 2]
    df10 = df9.drop(['size', 'price_per_sqft'], axis='columns')
    dummies = pd.get_dummies(df10.location)
    df11 = pd.concat([df10, dummies.drop('other', axis='columns')], axis='columns')
    df12 = df11.drop('location', axis='columns')
    a = df12.drop(['price'], axis='columns')
    y = df12.price
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(a, y, test_size=0.2, random_state=10)
    from sklearn.linear_model import LinearRegression
    lr_clf = LinearRegression()
    lr_clf.fit(x_train, y_train)

    def predict_price(location, sqft, bath, bhk):
        loc_index = np.where(a.columns == location)[0][0]

        x = np.zeros(len(a.columns))
        x[0] = sqft
        x[1] = bath
        x[2] = bhk
        if loc_index >= 0:
            x[loc_index] = 1

        return lr_clf.predict([x])[0]

    var1 = str(request.GET['n1'])
    var2 = float(request.GET['n2'])
    var3 = float(request.GET['n3'])
    var4 = float(request.GET['n4'])

    pred = predict_price(var1, var2, var3, var4)
    pred = round(pred)

    price = "The predicted price is Rs."+str(pred)+"(amount in lakh)"
    return render(request, "predict.html", {"result2": price})