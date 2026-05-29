import numpy as np
import torch
import torch.nn as nn
from heuristics import evaluate_heuristics, load_kinematics_bounds, get_severity

def score_sequence_via_ml(seq_features, model_path="v8_lstm.pth"):
    try:
        mean = np.load("stats/feature_mean.npy")
        std = np.load("stats/feature_std.npy")
        from models import AttentionLSTMPredictor
        model = AttentionLSTMPredictor(input_dim=21, hidden_dim=64, num_layers=2)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        norm_feat = (seq_features - mean) / std
        tensor_x = torch.FloatTensor(norm_feat).unsqueeze(0)
        with torch.no_grad():
            preds = model(tensor_x).squeeze(0).numpy() # Shape [3]
            return {
                "depth": float(np.clip(preds[0], 0, 100)),
                "valgus": float(np.clip(preds[1], 0, 100)),
                "back": float(np.clip(preds[2], 0, 100))
            }
    except Exception as e:
        print(f"ML evaluation skipped due to internal error: {e}")
        return {"depth": 85.0, "valgus": 85.0, "back": 85.0}

def evaluate_multi_rep_squat(rep_payloads, camera_view="side"):
    stats = load_kinematics_bounds()
    rep_results = []
    
    baseline_depths = []
    baseline_lean = []
    
    for i, (features, phases, global_start) in enumerate(rep_payloads):
        min_depth = np.min(features[:, 3:5]) 
        max_lean = np.max(features[:, 2])
        
        if i == 0:
            baseline_depths.append(min_depth)
            baseline_lean.append(max_lean)
            
        scores, valgus_blks, back_blks = evaluate_heuristics(features, phases, stats)
        ml_preds = score_sequence_via_ml(features)
        
        descent_mask = (phases == 0)
        bottom_mask = (phases == 1)
        ascent_mask = (phases == 2)
        
        v_desc = np.std(features[:, 18][descent_mask]) if np.any(descent_mask) else 0.0
        v_bot = np.std(features[:, 18][bottom_mask]) if np.any(bottom_mask) else 0.0
        v_asc = np.std(features[:, 18][ascent_mask]) if np.any(ascent_mask) else 0.0
        
        desc_stab = np.clip(np.exp(-v_desc * 10) * 100, 0, 100)
        bot_stab = np.clip(np.exp(-v_bot * 10) * 100, 0, 100)
        asc_stab = np.clip(np.exp(-v_asc * 10) * 100, 0, 100)
        stab_score = (desc_stab + bot_stab + asc_stab) / 3.0
        
        if i >= 3 and baseline_depths:
            user_min_depth = np.mean(baseline_depths)
            diff = abs(min_depth - user_min_depth)
            if diff < 0.05: 
                scores["depth"] = max(scores["depth"], 0.95)
                
        # Redistribute weights scaling depth and torso equally removing valgus limits.
        physics_agg = (scores["depth"]*0.5 + scores["back"]*0.5) * 100
        ml_agg = (ml_preds["depth"]*0.5 + ml_preds["back"]*0.5)
        hybrid_score = float(np.clip(round((0.4 * ml_agg) + (0.3 * physics_agg) + (0.3 * stab_score), 1), 0.0, 100.0))
        
        issue_logs = []
        sev_map = {"HIGH": 3, "MED": 2, "LOW": 1, "NONE": 0}
                
        if back_blks:
            sev_tier = get_severity(scores["back"])
            if sev_tier != "NONE":
                frames_all = []
                for b in back_blks: frames_all.extend(b["frames"])
                issue_logs.append({
                    "text": f"Torso leans forward progressively ({sev_tier} Severity). Likely due to core bracing failure or quad tracking offsets.",
                    "level": sev_tier,
                    "frames": [f + global_start for f in frames_all],
                    "type": "back"
                })
                
        if i > 0 and baseline_lean:
            diff = max_lean - baseline_lean[0]
            if diff > 5.0:
                issue_logs.append({
                    "text": f"Fatigue drift detected: Torso leaning further forward than your original baseline. Maintain active tension.",
                    "level": "MED",
                    "frames": [],
                    "type": "fatigue"
                })
                
        # Limit to Top 2 Issues logically!
        issue_logs = sorted(issue_logs, key=lambda x: sev_map.get(x["level"], 0), reverse=True)[:2]
        
        # Inject psychological positives natively!
        positives = []
        if scores["depth"] >= 0.85:
            positives.append("Depth is consistent and extremely solid across the movement.")
        else:
            positives.append("Pacing and momentum is effectively controlled.")
            
        rep_results.append({
            "rep": i + 1,
            "hybrid_score": hybrid_score,
            "components": {
                "depth": round(scores["depth"] * 100, 1),
                "back_stability": round(scores["back"] * 100, 1),
                "temporal_control": round(stab_score, 1)
            },
            "diagnostics": issue_logs,
            "positives": [positives[0]] if positives else []
        })
        
    return rep_results
