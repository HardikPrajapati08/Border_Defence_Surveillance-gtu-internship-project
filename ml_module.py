import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import random

def generate_simulated_data(num_records=500):
    """
    Generates simulated historical data for border intrusions.
    """
    np.random.seed(42)
    
    classes = ['Human', 'Vehicle', 'Animal']
    sectors = ['Alpha', 'Bravo', 'Charlie', 'Delta']
    
    data = []
    
    for _ in range(num_records):
        obj_class = random.choices(classes, weights=[0.4, 0.2, 0.4])[0]
        
        # Velocity in km/h
        if obj_class == 'Human':
            velocity = round(np.random.normal(5, 2), 2)
        elif obj_class == 'Vehicle':
            velocity = round(np.random.normal(40, 15), 2)
        else:
            velocity = round(np.random.normal(15, 10), 2)
            
        velocity = max(0.1, velocity)
            
        # time_in_zone_sec
        time_in_zone = max(0, int(np.random.normal(20, 15))) if obj_class != 'Animal' else int(np.random.normal(60, 30))
        
        # Time of day (0-23 hours)
        time_of_day = int(np.random.uniform(0, 24))
        
        # Add spatial coordinates for heatmap (x, y) 0-1000
        x_coord = np.random.randint(0, 1000)
        y_coord = np.random.randint(0, 1000)
        
        sector = random.choice(sectors)
        
        # Create some intentional anomalies (e.g., human very slow at 3 AM in sector Bravo)
        is_anomaly_injected = random.random() < 0.05
        if is_anomaly_injected:
            obj_class = 'Human'
            time_of_day = random.choice([1, 2, 3])
            velocity = round(random.uniform(0.5, 1.5), 2) # Creeping
            time_in_zone = random.randint(120, 300) # Loitering a long time
            sector = 'Bravo'
            x_coord = np.random.randint(400, 600) # Cluster anomalies
            y_coord = np.random.randint(400, 600)
            
        data.append({
            'object_class': obj_class,
            'velocity_kmh': velocity,
            'time_in_zone_sec': time_in_zone,
            'time_of_day': time_of_day,
            'sector': sector,
            'x_coord': x_coord,
            'y_coord': y_coord,
            'injected_anomaly': is_anomaly_injected
        })
        
    return pd.DataFrame(data)

def train_anomaly_model(df):
    """
    Trains an Isolation Forest on the data to detect behavioral anomalies.
    Returns the dataframe with a new column 'threat_level' (0 or 1).
    """
    # Features for the model
    features = ['velocity_kmh', 'time_in_zone_sec', 'time_of_day']
    X = df[features]
    
    # Initialize Isolation Forest
    clf = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    
    # Fit and predict (-1 is anomaly, 1 is normal)
    preds = clf.fit_predict(X)
    
    # Map -1 to 1 (Threat) and 1 to 0 (Normal)
    df['threat_level'] = [1 if p == -1 else 0 for p in preds]
    
    return df, clf
