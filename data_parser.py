import numpy as np
import json
import os
import glob

JOINTS = {
    'Nose': 0, 'Neck': 1, 'RShoulder': 2, 'RElbow': 3, 'RWrist': 4,
    'LShoulder': 5, 'LElbow': 6, 'LWrist': 7, 'MidHip': 8, 'RHip': 9,
    'RKnee': 10, 'RAnkle': 11, 'LHip': 12, 'LKnee': 13, 'LAnkle': 14,
    'REye': 15, 'LEye': 16, 'REar': 17, 'LEar': 18, 'LBigToe': 19,
    'LSmallToe': 20, 'LHeel': 21, 'RBigToe': 22, 'RSmallToe': 23, 'RHeel': 24
}

def remove_outliers(sequence, threshold=1.0):
    """ Reject sudden pose jumps via spatial derivative thresholding """
    T, J, D = sequence.shape
    cleaned = np.copy(sequence)
    for t in range(1, T-1):
        diff = np.linalg.norm(cleaned[t] - cleaned[t-1], axis=1) # [25]
        jumps = diff > threshold
        if np.any(jumps):
            # Linearly interpolate missing anomalous jumps
            cleaned[t, jumps] = (cleaned[t-1, jumps] + cleaned[t+1, jumps]) / 2.0
    return cleaned

def time_warp_sequence(sequence, target_length=100):
    """ Warps sequence identically to a standard timeframe (100 frames) for LSTM inputs """
    T, J, D = sequence.shape
    if T == target_length: return sequence
    old_indices = np.linspace(0, 1, T)
    new_indices = np.linspace(0, 1, target_length)
    warped = np.zeros((target_length, J, D))
    for j in range(J):
        for d in range(D):
            warped[:, j, d] = np.interp(new_indices, old_indices, sequence[:, j, d])
    return warped

def normalize_pose_3d(sequence):
    """
    V5 Scale / Rotate rigid canonical mapper.
    Standardizes sequence length and fully scrubs outliers cleanly.
    """
    # Layer 1: Outlier rejection & Sequence Warp Standardization
    seq = remove_outliers(sequence, threshold=1.5)
    seq = time_warp_sequence(seq, target_length=100) 
    
    T, num_joints, dims = seq.shape
    norm_seq = np.zeros_like(seq)
    
    for t in range(T):
        frame = seq[t]
        
        # 1. Translate -> Origin is now MidHip
        r_hip = frame[JOINTS['RHip']]
        l_hip = frame[JOINTS['LHip']]
        mid_hip = (r_hip + l_hip) / 2.0
        centered = frame - mid_hip
        
        # 2. Scale -> Torso length normalized
        neck = centered[JOINTS['Neck']]
        torso_len = np.linalg.norm(neck) + 1e-6
        scaled = centered / torso_len
        
        # 3. Rotate to canonical frame (X: lateral, Y: vertical, Z: forward)
        v_axis = scaled[JOINTS['Neck']]
        v_axis = v_axis / (np.linalg.norm(v_axis) + 1e-6)
        
        hip_line = scaled[JOINTS['LHip']] - scaled[JOINTS['RHip']]
        hip_line = hip_line / (np.linalg.norm(hip_line) + 1e-6)
        
        f_axis = np.cross(v_axis, hip_line)
        f_axis = f_axis / (np.linalg.norm(f_axis) + 1e-6)
        
        l_axis = np.cross(f_axis, v_axis)
        l_axis = l_axis / (np.linalg.norm(l_axis) + 1e-6)
        
        R = np.vstack([l_axis, v_axis, f_axis])
        rotated = np.dot(scaled, R.T)
        norm_seq[t] = rotated
        
    return norm_seq

def prepare_dataset(base_dir):
    dataset = []
    subject_dirs = glob.glob(os.path.join(base_dir, "s*"))
    for sdir in subject_dirs:
        squat_file = os.path.join(sdir, "joints3d_25", "squat.json")
        ann_file = os.path.join(sdir, "rep_ann.json")
        if not os.path.exists(squat_file) or not os.path.exists(ann_file):
            continue
            
        with open(squat_file, 'r') as f:
            seq_dict = json.load(f)
            joints_3d = np.array(seq_dict['joints3d_25']) # [T, 25, 3]
            
        with open(ann_file, 'r') as f:
            ann = json.load(f)
            
        squat_reps = ann.get('squat', [])
        if len(squat_reps) > 1:
            for i in range(len(squat_reps) - 1):
                try:
                    start = int(squat_reps[i])
                    end = int(squat_reps[i+1])
                    if end - start > 10:
                        rep_seq = joints_3d[start:end+1]
                        dataset.append(normalize_pose_3d(rep_seq))
                except Exception:
                    continue
                
    return dataset
