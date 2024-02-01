from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def preprocessing(filepath):
    """
    Loads and preprocesses the data
    """
    data = pd.read_csv(filepath)
    # Convert TIMESTAMP to datetime and extract hours
    data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP'])
    data['hour'] = data['TIMESTAMP'].dt.hour
    data['weekday'] = data['TIMESTAMP'].dt.weekday
    data['dayofyear']=data['TIMESTAMP'].dt.day_of_year
    
    data['CLOUD_COVERAGE'] = data['CLOUD_COVERAGE']/100
    
    data['week'] = data['dayofyear']//7 -(data['dayofyear']//7).min()

    # Scale user and venue LONG and LAT data
    # Initialize the scaler
    MINMAXscaler= MinMaxScaler()
    scaled_data = MINMAXscaler.fit_transform(data[['USER_LONG', 'VENUE_LONG','USER_LAT', 'VENUE_LAT','PRECIPITATION','CLOUD_COVERAGE','TEMPERATURE','WIND_SPEED']])
    data['USER_LONG_scaled'] = scaled_data[:, 0]
    data['VENUE_LONG_scaled'] = scaled_data[:, 1]
    data['USER_LAT_scaled'] = scaled_data[:, 2]
    data['VENUE_LAT_scaled'] = scaled_data[:, 3]
    data['PRECIPITATION_scaled'] = scaled_data[:,4]
    data['CLOUD_COVERAGE_scaled'] = scaled_data[:,5]
    data['TEMPERATURE_scaled'] = scaled_data[:,6]
    data['WIND_SPEED_scaled'] = scaled_data[:,7]


    scaled_users = data[['USER_LAT_scaled', 'USER_LONG_scaled']]
    scaled_venues = data[['VENUE_LAT_scaled', 'VENUE_LONG_scaled']]
    # Compute l2 distance between users and venues
    data['scaled_distance'] = np.linalg.norm(scaled_users.values - scaled_venues.values, axis=1)
    
    return data

def add_weekday_hour(DATASET):
    if 'weekday_hour' not in DATASET.columns:
        day_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5:'Saturday', 6:'Sunday'}
        DATASET['weekday_hour'] = DATASET['weekday'].map(day_names) + '_' + DATASET['hour'].astype(str)
        return DATASET, day_names
    else:
        raise('weekday_hour already in columns')

def plot_average_week(DAILY_ORDERS):

    DAILY_ORDERS, day_names = add_weekday_hour(DAILY_ORDERS)

    # Plot settings
    plt.figure(figsize=(15, 6))
    plt.title(f"Average Number of Orders during each Day of the Week", fontsize=20)
    plt.ylabel('No. of Orders', fontsize=16)
    plt.xlabel('Weekday and Hour', fontsize=16)

    # Plot the data
    sns.lineplot(x='weekday_hour', y='orders', data=DAILY_ORDERS, estimator=np.mean, errorbar=('ci', 95), label='Orders')

    # Set custom x-axis ticks
    # Ensure custom ticks align with the actual data
    custom_ticks = [f"{day}_12" for day in day_names.values()]  # '12' represents 12 PM in 24-hour format
   
    plt.xticks(custom_ticks, rotation=45)

    # Final adjustments and show plot
    plt.tight_layout()
    plt.show()

def check_unique(df):
    """
    Used for checking how often the weather data is updated (answer: every hour)    
    """
    for col in df.columns:
        if df[col].nunique() != 1:
            raise ValueError('Column {} has more than one unique value'.format(col))
        
def add_padding(OBSERVED_DATA: pd.DataFrame, ADD_DAY_HOUR_FEATURE: bool = True):
    """
    Adds padding of unobserved hours during the days included in OBSERVED_DATA.
    Weather, venue_cluster, weekday are backward and forward filled. Orders are filled with 0.
    Returns:
        OBSERVED_DATA: DataFrame with padded data
    """
    actual_data = OBSERVED_DATA.copy()
    all_hours = np.arange(24)
    all_days = actual_data['dayofyear'].unique()
    all_venue_clusters = actual_data['venue_cluster'].unique()

    multi_index = pd.MultiIndex.from_product([all_days, all_hours, all_venue_clusters], 
                                            names=['dayofyear', 'hour', 'venue_cluster'])

    # Reindex the DataFrame
    padded_df = actual_data.set_index(['dayofyear', 'hour', 'venue_cluster']).reindex(multi_index)

    # Forward and backward fill for specific columns
    padded_df[['weekday', 'TEMPERATURE_scaled', 'WIND_SPEED_scaled', 'PRECIPITATION_scaled', 'CLOUD_COVERAGE_scaled']] = \
        padded_df[['weekday', 'TEMPERATURE_scaled', 'WIND_SPEED_scaled', 'PRECIPITATION_scaled', 'CLOUD_COVERAGE_scaled']].ffill().bfill()

    # Set 'orders' to 0 for padded rows
    padded_df['orders'].fillna(0, inplace=True)

    # Reset index to make 'dayofyear' and 'hour' columns again
    padded_df.reset_index(inplace=True)
    if ADD_DAY_HOUR_FEATURE:
        padded_df['day_hour'] = padded_df['dayofyear'].astype(str) + '_' + padded_df['hour'].astype(str)

    return padded_df
        
def augment_ohe_clusters(DATASET):
    """
    DATASET: df with categorical variable 'venue_cluster' to be o-h-e
    """
    ohe_clusters = pd.get_dummies(DATASET['venue_cluster'], prefix='venue_cluster')
    concat_DATASET = pd.concat([DATASET, ohe_clusters], axis=1)
    return concat_DATASET.drop('venue_cluster', axis=1)

def scale_NN_input(INPUT,PRETRAINED_SCALER):
    # Prepare data for input in neural network, using pretrained scaler
    ohe_input = augment_ohe_clusters(INPUT)
    scaled_input = PRETRAINED_SCALER.transform(ohe_input)
    return scaled_input

def plot_test_loss(model_history):
    train_loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']

    # Creating a range object for the number of epochs
    epochs = range(1, len(train_loss) + 1)

    # Plotting the training and validation loss
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_loss, 'bo', label='Training loss')  # 'bo' gives us blue dots
    plt.plot(epochs, val_loss, 'b', label='Validation loss')  # 'b' gives us a solid blue line
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Create 'orders' column: counts the number of orders (for each hour, cluster and weather conditions)
def count_orders(data_set):
    counted_set = data_set.groupby(data_set.columns.tolist()).size().reset_index(name='orders')
    return counted_set

def get_test_pred_df(TEST_SET,SET_PREDICTION):
    predicted_data = pd.concat([TEST_SET.drop(['orders'],axis=1),pd.DataFrame(SET_PREDICTION,columns=['orders'])],axis=1)
    predicted_data['day_hour'] = predicted_data['dayofyear'].astype(str) + '_' + predicted_data['hour'].astype(str)
    return predicted_data

def plot_test_prediction(PADDED_DATA,TEST_PREDICTION, TEST_DAYS,ALGORITHM_NAME):
    plt.figure(figsize=(8, 4))
    # Line plot for actual data
    if 'day_hour' not in PADDED_DATA.columns:
        PADDED_DATA['day_hour'] = PADDED_DATA['dayofyear'].astype(str) + '_' + PADDED_DATA['hour'].astype(str)
    sns.lineplot(x='day_hour', y='orders', data=PADDED_DATA, marker='o', label='Observed')
    # Scatter plot for predicted data
    if ALGORITHM_NAME == 'MLP':
        color = 'red'
    elif ALGORITHM_NAME == 'RNN':
        color = 'magenta'
    elif ALGORITHM_NAME == 'RF':
        color = 'orange'
    elif ALGORITHM_NAME == 'KNN':
        color = 'purple'
    elif ALGORITHM_NAME == 'XGB':
        color = 'brown'

    sns.lineplot(x='day_hour', y='orders', data=TEST_PREDICTION, color=color,label=f'{ALGORITHM_NAME}')

    # Custom ticks for better readability
    custom_ticks = [f"{day}_12" for day in TEST_DAYS]
    plt.xticks(custom_ticks, rotation=45)

    # Improve the plot aesthetics
    plt.xlabel('Day_Hour')
    plt.ylabel('Number of Orders')
    plt.title(f'Observed vs Predicted Average of Orders by Day-Hour using {ALGORITHM_NAME}')
    plt.legend(title='Data Type', loc='upper left')

    # Show the plot
    plt.show()

def plot_test_scatter(PADDED_DATA,TEST_PREDICTION1,TEST_PREDICTION2, TEST_DAYS):
    plt.figure(figsize=(12, 6))
    # Line plot for actual data
    if 'day_hour' not in PADDED_DATA.columns:
        PADDED_DATA['day_hour'] = PADDED_DATA['dayofyear'].astype(str) + '_' + PADDED_DATA['hour'].astype(str)
    sns.scatterplot(x='day_hour', y='orders', data=PADDED_DATA,color='blue', marker='o', label='Observed')
    # Scatter plot for predicted data

    sns.scatterplot(x='day_hour', y='orders', data=TEST_PREDICTION1, color='orange',label='RF')
    sns.scatterplot(x='day_hour', y='orders', data=TEST_PREDICTION2, color='brown',label='XGB')

    # Custom ticks for better readability
    custom_ticks = [f"{day}_12" for day in TEST_DAYS]
    plt.xticks(custom_ticks, rotation=45)

    # Improve the plot aesthetics
    plt.xlabel('Day_Hour')
    plt.ylabel('Number of Orders')
    plt.title(f'Observed vs Predicted Orders by Day-Hour using RF and XGB')
    plt.legend(title='Data Type', loc='upper left')

    # Show the plot
    plt.show()