import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
# from joblib import dump
import joblib
from joblib import load
test_data = pd.read_csv('scoring.csv').drop(['Unnamed: 0'], axis=1)
import math

def calculate_direction(x1, y1, x2, y2):
    delta_x = x2 - x1
    delta_y = y2 - y1
    angle_radians = math.atan2(delta_y, delta_x)
    angle_degrees = math.degrees(angle_radians)
    return angle_radians, angle_degrees

test_data = pd.read_csv('scoring.csv').drop(['Unnamed: 0'], axis=1)

test_data['angle'] = test_data.apply(lambda x: calculate_direction(x['bh_x'], x['bh_y'], x['horizontal_toe_x'], x['horizontal_toe_y'])[0], axis=1)
test_data['proppant_per_stage'] = test_data['total_proppant'] / test_data['number_of_stages']
test_data['fluid_per_stage'] = test_data['total_fluid'] / test_data['number_of_stages']
test_data['proppant_intensity'] = test_data['total_proppant'] / test_data['gross_perforated_length']
test_data['fluid_intensity'] = test_data['total_fluid'] / test_data['gross_perforated_length']


categorical_columns = ['ffs_frac_type', 'relative_well_position', 'batch_frac_classification',
       'well_family_relationship', ]

for column in categorical_columns:
    # Load the encoder
    print(column)
    le = load(f'label_encoders/{column}_encoder.joblib')
    
    # Check if the column exists in the new data
    if column in test_data.columns:
        
        test_data[column] = le.transform(test_data[column])
    else:
        # Handle the case where the column does not exist in the new data
        print(f"Column '{column}' not found in new data.")

# Now 'new_data' is ready for inference with your model
cols = np.load('model_files/columns.npy')
final_test_data = test_data[cols]

scalar = joblib.load('model_files/scaler.joblib')
best_model = joblib.load('model_files/best_model.joblib')

features = scalar.transform(final_test_data.values)
rf_predictions = best_model.predict(features)

sub = pd.DataFrame({'OilPeakRate': rf_predictions})