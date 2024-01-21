import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

from sklearn.preprocessing import StandardScaler

df = pd.read_csv('../first/training.csv').drop(columns=['Unnamed: 0'])




#Remove certain outliers. total_proppant, total_fluid, proppant_i|ntensity with quantile 0.99 for df_well
df = df[df['total_fluid'] < df['total_fluid'].quantile(0.99)]
print(df.shape)
df = df[df['total_proppant'] < df['total_proppant'].quantile(0.99)]
print(df.shape)
df = df[df['proppant_intensity'] < df['proppant_intensity'].quantile(0.99)]
print(df.shape)
df = df[df['proppant_to_frac_fluid_ratio'] < df['proppant_to_frac_fluid_ratio'].quantile(0.99)]
print(df.shape)
df = df[df['frac_fluid_to_proppant_ratio'] < df['frac_fluid_to_proppant_ratio'].quantile(0.994)]
# print(df.shape)
df = df[df['OilPeakRate'] < df['OilPeakRate'].quantile(0.99)]
print(df.shape)

### Remove outliers for other columns
for col in df.columns:
    s2 = ['total_fluid', 'total_proppant', 'proppant_intensity', 'proppant_to_frac_fluid_ratio', 'frac_fluid_to_proppant_ratio', 'OilPeakRate']
    if col in s2:
        df = df[df[col] < df[col].quantile(0.995)]

df = df[df['bin_lateral_length'] >=1]

print(df.shape)
df.dropna(subset=['OilPeakRate'], inplace=True)
print(df.shape)

print(df.info())



## Filling number of stages
df_to_fill = df[['bh_x', 'bh_y', 'horizontal_toe_x', 'horizontal_toe_y', 'horizontal_midpoint_x', 'horizontal_midpoint_y', 'bin_lateral_length','number_of_stages','total_fluid']]

# Create train set, all non nan

df_to_train = df_to_fill.dropna()
# df_to_train



## Training imputation model
features = df_to_train.drop(columns=['number_of_stages'])
target =  df_to_train['number_of_stages']



scaler = StandardScaler()
features = scaler.fit_transform(features.values)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
# Create the Random Forest model
rf_model = RandomForestRegressor(random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Predict using tahe model
rf_predictions = rf_model.predict(X_test)

# Evaluate the model
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_rmse = np.sqrt(rf_mse)
rf_r2_score = r2_score(y_test, rf_predictions)

print("Random Forest RMSE:", rf_rmse)
print("Random Forest R2 Score:", rf_r2_score)

# Imputing number of stages
def predict_number_of_stages(row):
    if not pd.isna(row['number_of_stages']):
        return row['number_of_stages']
    else:
        row = row[['bh_x', 'bh_y', 'horizontal_toe_x', 'horizontal_toe_y', 'horizontal_midpoint_x', 'horizontal_midpoint_y', 'bin_lateral_length','total_fluid']]
        if row.isna().sum() > 0:
            return np.nan
        else:
            # print(row.values)
           
            row = scaler.transform(row.values.reshape(1, -1))
            out = rf_model.predict(row.reshape(1, -1))[0]
           
            return out
    
df['number_of_stages'] = df.apply(predict_number_of_stages, axis=1)


# Fill in missing values for average_stage_length
df_to_fill = df[['bh_x', 'bh_y', 'horizontal_toe_x', 'horizontal_toe_y', 'horizontal_midpoint_x', 'horizontal_midpoint_y', 'bin_lateral_length','number_of_stages','total_fluid','average_stage_length' ]]



features = df_to_train.drop(columns=['average_stage_length'])
target =  df_to_train['average_stage_length']


# add standard scalar
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features = scaler.fit_transform(features.values)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
# Create the Random Forest model
rf_model = RandomForestRegressor(random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Predict using tahe model
rf_predictions = rf_model.predict(X_test)

# Evaluate the model
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_rmse = np.sqrt(rf_mse)
rf_r2_score = r2_score(y_test, rf_predictions)

print("Random Forest RMSE:", rf_rmse)
print("Random Forest R2 Score:", rf_r2_score)

def predict_avg_prop(row):
    if not pd.isna(row['average_stage_length']):
        return row['average_stage_length']
    else:
        row = row[['bh_x', 'bh_y', 'horizontal_toe_x', 'horizontal_toe_y', 'horizontal_midpoint_x', 'horizontal_midpoint_y', 'bin_lateral_length','number_of_stages','total_fluid']]
        if row.isna().sum() > 0:
            return np.nan
        else:
            # print(row.values)
            row = scaler.transform(row.values.reshape(1, -1))
            out = rf_model.predict(row.reshape(1, -1))[0]
           
            return out
    
df['average_stage_length'] = df.apply(predict_avg_prop, axis=1)


#### Drop 'average_proppant_per_stage','average_frac_fluid_per_stage'

df= df.drop(columns=['average_proppant_per_stage','average_frac_fluid_per_stage'])




## Label Encoding
import pandas as pd
from sklearn.preprocessing import LabelEncoder


categorical_columns = ['ffs_frac_type', 'relative_well_position', 'batch_frac_classification',
       'well_family_relationship', 'frac_type','ffs_frac_type']

# Initialize LabelEncoder
le = LabelEncoder()

# Apply LabelEncoder to each categorical column
for column in categorical_columns:
    df[column] = le.fit_transform(df[column])
df.to_csv('before_final_imp.csv')

# Fill rest
df.fillna(df.mean(), inplace=True)
df.to_csv('final_imp.csv')


## Single model with 0.2 test size
features = df.drop(columns=['OilPeakRate'])
target = df['OilPeakRate']

scaler = StandardScaler()
features = scaler.fit_transform(features.values)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
# Create the Random Forest model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Predict using tahe model
rf_predictions = rf_model.predict(X_test)

# Evaluate the model
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_rmse = np.sqrt(rf_mse)
rf_r2_score = r2_score(y_test, rf_predictions)

print("Random Forest RMSE:", rf_rmse)
print("Random Forest R2 Score:", rf_r2_score)


# Calculate residuals
from scipy import stats


residuals = y_test - rf_predictions

# Plotting the residuals distribution
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, bins=30)
plt.title('Residuals Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

# Q-Q plot for normality check
plt.figure(figsize=(10, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot for ')
plt.show()

## Cross validation

features = df.drop(columns=['OilPeakRate'])
target = df['OilPeakRate']

# Standard scaling
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features.values)

# Random Forest model
rf_model = RandomForestRegressor(random_state=42, n_jobs=12)

# Cross-validation (using R2 as the metric)
cv_scores = cross_val_score(rf_model, features_scaled, target, cv=10, scoring='neg_root_mean_squared_error', n_jobs=12)

# Calculate average R2 score across all folds
average_r2_score = np.mean(cv_scores)
print("Average R2 Score: Base Model", average_r2_score)




to_drop_col= ['proppant_to_frac_fluid_ratio', 'bh_x', 'bh_y', 'horizontal_midpoint_x', 'well_family_relationship', 'proppant_to_frac_fluid_ratio']
df_final = df.drop(columns=to_drop_col)



features = df_final.drop(columns=['OilPeakRate'])
target =  df_final['OilPeakRate']


# Standard scaling
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features.values)

# Random Forest model
rf_model = RandomForestRegressor(random_state=42, n_jobs=12)

# Cross-validation (using R2 as the metric)
cv_scores = cross_val_score(rf_model, features_scaled, target, cv=10, scoring='neg_root_mean_squared_error', n_jobs=12)

# Calculate average R2 score across all folds
average_r2_score = np.mean(cv_scores)
print("Average R2 Score: feature engineered Model", average_r2_score)

##########################################

# add standard scalar
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features = scaler.fit_transform(features.values)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.1, random_state=42)
# Create the Random Forest model
rf_model = RandomForestRegressor(random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Predict using tahe model
rf_predictions = rf_model.predict(X_test)

# Evaluate the model
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_rmse = np.sqrt(rf_mse)
rf_r2_score = r2_score(y_test, rf_predictions)

print("Random Forest RMSE:", rf_rmse)
print("Random Forest R2 Score:", rf_r2_score)


## Adding Features
import math

def calculate_direction(x1, y1, x2, y2):
    delta_x = x2 - x1
    delta_y = y2 - y1
    angle_radians = math.atan2(delta_y, delta_x)
    angle_degrees = math.degrees(angle_radians)
    return angle_radians, angle_degrees

df['angle'] = df.apply(lambda x: calculate_direction(x['bh_x'], x['bh_y'], x['horizontal_toe_x'], x['horizontal_toe_y'])[0], axis=1)
df['proppant_per_stage'] = df['total_proppant'] / df['number_of_stages']
df['fluid_per_stage'] = df['total_fluid'] / df['number_of_stages']
df['proppant_intensity'] = df['total_proppant'] / df['gross_perforated_length']
df['fluid_intensity'] = df['total_fluid'] / df['gross_perforated_length']


features = df.drop(columns=['OilPeakRate'])
target =  df['OilPeakRate']


# add standard scalar
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
# features = scaler.fit_transform(features.values)
features = np.log1p(features.values)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.1, random_state=42)
from sklearn.model_selection import cross_val_score
features = data.drop(columns=['OilPeakRate'])
target = data['OilPeakRate']

# Standard scaling
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features.values)

# Random Forest model
rf_model = RandomForestRegressor(random_state=42)

# Cross-validation (using R2 as the metric)
cv_scores = cross_val_score(rf_model, features_scaled, target, cv=10, scoring='neg_root_mean_squared_error')

# Calculate average R2 score across all folds
average_r2_score = np.mean(cv_scores)
print("Average R2 Score: with added features", average_r2_score)