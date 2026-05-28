import numpy as np

def load_kinematics_bounds():
    try:
        import json
        with open("fit3d_kinematics.json", "r") as f:
            return json.load(f)
    except:
        return {
            "valgus_max": [10.0, 5.0]
        }

def get_severity(score):
    if score >= 0.85: return "NONE"
    elif score >= 0.70: return "LOW"
    elif score >= 0.50: return "MED"
    return "HIGH"

def aggregate_phase_errors(active_frames, phases, issue_name):
    """
    V6 Core Node: Abandons frame-level polling in favor of strict temporal aggregation loops.
    Groups anomalous slices logically and computes bounds strictly relative to internal phase tracking endpoints.
    """
    if not np.any(active_frames): return []
    
    blocks = []
    current_block = []
    
    for t, active in enumerate(active_frames):
        if active:
            if current_block and phases[t] != phases[current_block[-1]]:
                if len(current_block) > 3:
                    blocks.append(current_block)
                current_block = [t]
            else:
                current_block.append(t)
        else:
            if len(current_block) > 3:
                blocks.append(current_block)
            current_block = []
    if len(current_block) > 3:
        blocks.append(current_block)
        
    results = []
    T = len(active_frames)
    for b in blocks:
        start, end = b[0], b[-1]
        
        p_slice = phases[start:end+1]
        p_mode = int(np.bincount(p_slice).argmax())
        phase_map = {0: "descent", 1: "bottom", 2: "ascent"}
        p_name = phase_map.get(p_mode, "descent")
        
        phase_indices = np.where(phases == p_mode)[0]
        if len(phase_indices) > 0:
            phase_start = phase_indices[0]
            phase_end_len = len(phase_indices)
            start_pct = int(((start - phase_start) / phase_end_len) * 100)
            end_pct = int(((end - phase_start) / phase_end_len) * 100)
        else:
            start_pct, end_pct = 0, 100
            
        start_pct = max(0, min(100, start_pct))
        end_pct = max(0, min(100, end_pct))
        
        results.append({
            "issue": issue_name,
            "phase": p_name,
            "start_pct": start_pct,
            "end_pct": end_pct,
            "frames": b
        })
    return results

def evaluate_heuristics(features, phases, bounds, camera_view="side"):
    """ Natively executes structural bounds mappings computing raw offset heuristics safely """
    scores = {"depth": 1.0, "back": 1.0}
    
    # 1. Depth Constraints (Y Axis Map)
    depth_arr = (features[:, 3] + features[:, 4]) / 2.0
    min_d = np.min(depth_arr)
    if min_d > 0.15:
        scores["depth"] = max(0.0, 1.0 - ((min_d - 0.15)/0.2)*0.5)
        
    # 2. Back / Torso Limits Tracker (Feature 2)
    back_arr = features[:, 2]
    back_mask = back_arr > 40.0
    back_blocks = aggregate_phase_errors(back_mask, phases, "forward_lean")
    
    max_b = np.max(back_arr)
    if max_b > 40.0:
        scores["back"] = max(0.0, 1.0 - ((max_b - 40.0)/20.0)*0.5)
        
    return scores, [], back_blocks
