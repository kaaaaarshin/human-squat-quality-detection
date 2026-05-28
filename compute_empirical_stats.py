import numpy as np
import json
from data_parser import prepare_dataset
from features import extract_features

def compute_fit3d_statistics(train_dir, out_file="fit3d_kinematics.json"):
    """
    Scans the entire Fit3D dataset to build dynamic empirical thresholds
    for optimal squat kinematics based purely on 3D data distributions.
    """
    print(f"Scanning dataset at {train_dir} to build empirical bounds...")
    dataset = prepare_dataset(train_dir)
    
    phase_data = {
        "descent": [],
        "bottom": [],
        "ascent": []
    }
    
    for seq in dataset:
        feats, phases = extract_features(seq)
        for t in range(len(feats)):
            p = phases[t]
            if p == 0:
                phase_data["descent"].append(feats[t])
            elif p == 1:
                phase_data["bottom"].append(feats[t])
            else:
                phase_data["ascent"].append(feats[t])
                
    stats = {}
    for p_name, data_list in phase_data.items():
        arr = np.array(data_list) # shape: [N, 14]
        if len(arr) == 0:
            continue
            
        stats[p_name] = {
            "mean": np.mean(arr, axis=0).tolist(),
            "std": np.std(arr, axis=0).tolist()
        }
        
    with open(out_file, 'w') as f:
        json.dump(stats, f, indent=4)
        
    print(f"Successfully computed empirical physics boundaries!")
    print(f"Saved statistics to {out_file}.")
    
    print("\n--- FIT3D 'BOTTOM' PHASE TARGETS ---")
    feat_names = ["R_Knee", "L_Knee", "BackAngle", "R_Depth", "L_Depth", "R_Valgus", "L_Valgus", "Knee_Sym", "Hip_Sym", "Lat_Stab", "Fwd_Stab", "Vel", "Acc", "Jerk"]
    
    if "bottom" in stats:
        means = stats["bottom"]["mean"]
        stds = stats["bottom"]["std"]
        for i, name in enumerate(feat_names):
            print(f"{name:10s} : Mean = {means[i]:7.2f}  |  Std = {stds[i]:7.2f}")

if __name__ == "__main__":
    train_dir = r"c:\Users\sriha\Desktop\ivp_proj\train"
    compute_fit3d_statistics(train_dir)
