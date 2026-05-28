import numpy as np
from data_parser import JOINTS

def smooth_sequence(sequence, alpha=0.3):
    """ Temporal Exponential Moving Average smoothing """
    smoothed = np.copy(sequence)
    for t in range(1, len(sequence)):
        smoothed[t] = alpha * sequence[t] + (1 - alpha) * smoothed[t-1]
    return smoothed

def calculate_angle_3d(v1, v2):
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

def extract_features(sequence):
    """
    V5 Feature Extraction Engine.
    Maps exactly 21 physical vectors across all sequences tracking precise 
    biomechanical mass distributions and stability limits.
    100 frame sequences -> [100, 21].
    """
    seq_3d = smooth_sequence(sequence, alpha=0.35)
    T = len(seq_3d)
    
    # Layer 2 Addition: Expansion to 21 total dimensions
    features = np.zeros((T, 21))
    base_hip_depths = []
    
    for t in range(T):
        frame = seq_3d[t]
        
        r_hip_knee = frame[JOINTS['RKnee']] - frame[JOINTS['RHip']]
        l_hip_knee = frame[JOINTS['LKnee']] - frame[JOINTS['LHip']]
        r_knee_ankle = frame[JOINTS['RAnkle']] - frame[JOINTS['RKnee']]
        l_knee_ankle = frame[JOINTS['LAnkle']] - frame[JOINTS['LKnee']]
        
        # New Feature: Ankle alignment / foot vector tracker computation
        r_ankle_foot = frame[JOINTS['RBigToe']] - frame[JOINTS['RAnkle']]
        l_ankle_foot = frame[JOINTS['LBigToe']] - frame[JOINTS['LAnkle']]
        
        spine = frame[JOINTS['Neck']] - frame[JOINTS['MidHip']]
        vertical_y = np.array([0.0, 1.0, 0.0])
        
        # 1. 3D Joint Angles
        r_knee_angle = calculate_angle_3d(r_hip_knee, r_knee_ankle)
        l_knee_angle = calculate_angle_3d(l_hip_knee, l_knee_ankle)
        back_angle = calculate_angle_3d(spine, vertical_y)
        
        # New Feature: Ankle angles explicitly 
        r_ankle_angle = calculate_angle_3d(r_knee_ankle, r_ankle_foot)
        l_ankle_angle = calculate_angle_3d(l_knee_ankle, l_ankle_foot)
        
        # 2. Depth Metrics (relative Y magnitude)
        r_hip_depth = frame[JOINTS['RHip']][1] - frame[JOINTS['RKnee']][1]
        l_hip_depth = frame[JOINTS['LHip']][1] - frame[JOINTS['LKnee']][1]
        avg_depth = (r_hip_depth + l_hip_depth) / 2.0
        base_hip_depths.append(avg_depth)
        
        # 3. 3D Valgus
        r_valgus_angle = calculate_angle_3d(r_hip_knee, -r_knee_ankle)
        l_valgus_angle = calculate_angle_3d(l_hip_knee, -l_knee_ankle)
        
        # 4. Symmetry
        r_hip_angle_3d = calculate_angle_3d(np.array([0.0, -1.0, 0.0]), r_hip_knee)
        l_hip_angle_3d = calculate_angle_3d(np.array([0.0, -1.0, 0.0]), l_hip_knee)
        knee_diff = abs(r_knee_angle - l_knee_angle)
        hip_diff = abs(r_hip_angle_3d - l_hip_angle_3d)
        
        # 5. Stability Shift
        lateral_stability = frame[JOINTS['MidHip']][0] 
        fwd_stability = frame[JOINTS['MidHip']][2]     
        
        # New Feature: COM (Center of Mass vs Foot midpoint)
        mid_foot_z = (frame[JOINTS['RAnkle']][2] + frame[JOINTS['RBigToe']][2] + 
                      frame[JOINTS['LAnkle']][2] + frame[JOINTS['LBigToe']][2]) / 4.0
        # COM approximation using MidHip bounds globally
        com_projection = frame[JOINTS['MidHip']][2] - mid_foot_z
        
        # New Feature: Femur Leverage ratio (hip to knee ratio compared against canonical 1.0 torso)
        hip_knee_ratio = (np.linalg.norm(r_hip_knee) + np.linalg.norm(l_hip_knee)) / 2.0
        
        features[t, :17] = [
            r_knee_angle, l_knee_angle, back_angle,
            r_hip_depth, l_hip_depth, r_valgus_angle, l_valgus_angle,
            knee_diff, hip_diff, lateral_stability, fwd_stability,
            r_ankle_angle, l_ankle_angle, com_projection, hip_knee_ratio,
            0.0, 0.0 # Placeholders for temporally calculated fields across vectors
        ]
        
    # Torso Stability Variance (Standard Dev over sequence mapped back to index arrays)
    spine_std = np.std(features[:, 2])
    features[:, 15] = spine_std
    
    # Velocity, Accel, Jerk temporal extractions natively mapped to the structure
    velocities = np.zeros(T)
    accelerations = np.zeros(T)
    jerks = np.zeros(T)
    
    for t in range(1, T):
        velocities[t] = base_hip_depths[t] - base_hip_depths[t-1]
    for t in range(1, T):
        accelerations[t] = velocities[t] - velocities[t-1]
    for t in range(1, T):
        jerks[t] = accelerations[t] - accelerations[t-1]
        
    features[:, 18] = velocities
    features[:, 19] = accelerations
    features[:, 20] = jerks
    
    # Dynamic inflection mapping (V7 Savitzky-Golay filtering)
    import scipy.signal
    phases = np.zeros(T, dtype=int)
    
    depth_coords = np.array(base_hip_depths)
    window = min(15, len(depth_coords))
    if window % 2 == 0: window -= 1
    
    if window > 3:
        smooth_depth = scipy.signal.savgol_filter(depth_coords, window, 3)
    else:
        smooth_depth = depth_coords
        
    bottom_frame = int(np.argmin(smooth_depth))
    
    descent_end = max(0, bottom_frame - 2)
    ascent_start = min(T-1, bottom_frame + 2)
    
    phases[:descent_end] = 0
    phases[descent_end:ascent_start] = 1
    phases[ascent_start:] = 2
    
    descent_time = max(descent_end, 1)
    ascent_time = max(T - ascent_start, 1)
    timing_ratio = float(descent_time) / float(ascent_time)
    features[:, 16] = timing_ratio
    
    return features, phases
