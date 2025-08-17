📂 Final Data Description

This folder contains three main datasets used for training, evaluation, visualization, and simulation in the project.

1. osrm.zip

File: drv_time_osrm_renamed.csv

Content:

Pre-computed shortest driving times from all fire stations to all incidents (calculated using OSRM).

Usage:

Provides baseline travel time data for routing, dispatch optimization, and simulation.

2. real_with_dispatch_info.zip

File: real_with_dispatch_info.csv

Content:

The main training dataset for reinforcement learning models (SAC and DQN).

Each row corresponds to a real incident dispatch record, including event attributes, station information, and dispatched vehicles.

Usage:

Training reinforcement learning models

Optimizing dispatching strategies

3. wmfs_mobilisations and station data.zip

Files:

wmfs_mobilisations.csv

station_mapping.pkl

station_xy.pkl

Station_engine_counts.csv

Content:

Mobilisation records (incident and vehicle dispatch details)

Fire station information (locations, mapping relationships)

Vehicle counts (number of appliances available at each station)

Usage:

Evaluation: Verify and validate model predictions

Visualization: Map-based analysis of incidents, stations, and dispatch flows

Simulation: Reproduce and analyze mobilisation processes under different scenarios

📌 Summary:

osrm.zip → Fire station–incident travel times (baseline traffic data)

real_with_dispatch_info.zip → Real dispatch dataset (RL training)

wmfs_mobilisations and station data.zip → Mobilisation records + station info + vehicle data (evaluation, visualization, simulation)
