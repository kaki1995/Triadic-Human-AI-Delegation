from pathlib import Path
import numpy as np, pandas as pd

base = Path(__file__).parent
input_xlsx = base / 'Triadic_Delegation_Dataset_SYNTH_ANALYSIS_v1.xlsx'
output_xlsx = base / 'Triadic_Delegation_Dataset_SYNTH_ANALYSIS_v2.xlsx'

xls = pd.ExcelFile(input_xlsx)
orig = {name: pd.read_excel(input_xlsx, sheet_name=name) for name in xls.sheet_names}
manager_master = orig['manager_master'].copy()
site_master = orig['site_master'].copy()

rng = np.random.default_rng(20260307)
periods = range(1, 27)
transparency_schedule = [0,1,2,3,2,1,0,2,3,2,1,0,2,3,2,1,0,2,3,2,1,0,2,3,2,1]
gov_orch_map = {'fearful_exclusion': -0.9, 'controlled_opening': 0.15, 'opportunistic_teaming': 0.95}
rows = []
for _, m in manager_master.iterrows():
    baseline = float(m['baseline_ai_attitude'])
    risk = float(m['risk_aversion_index'])
    hp = int(m['high_pressure'])
    gov = str(m['governance_mode'])
    site_id = m['site_id']
    site_complex = float(site_master.loc[site_master['site_id']==site_id, 'baseline_operational_complexity'].iloc[0])
    orch = gov_orch_map.get(gov, 0.0) + rng.normal(0, 0.20)
    heter = 0.65*baseline - 0.50*risk + 0.45*orch + rng.normal(0, 0.10)
    state = 1 if (baseline + 0.35*orch - 0.25*risk) > -0.05 else 0
    prev_perf = 0.0
    for p in periods:
        transparency = transparency_schedule[p-1]
        trans_eff = {0:-0.20,1:0.08,2:0.24,3:-0.16}[transparency]
        dv = float(np.clip(rng.beta(2.4,2.4)+0.06*hp+0.05*np.sin(p/3),0,1))
        tc = float(np.clip(rng.beta(2.3,2.1)+0.04*site_complex+0.03*np.cos(p/4),0,1))
        ts = float(np.clip(0.20+0.35*hp+0.22*dv+0.15*risk+rng.normal(0,0.05),0,1))
        shock = int(rng.random() < (0.05+0.13*dv+0.02*hp))
        rare = int(rng.random() < (0.03+0.10*dv+0.03*site_complex))
        amb = float(np.clip(0.12+0.45*rare+0.18*shock+0.18*dv+rng.normal(0,0.05),0,1))
        acc = float(np.clip(0.84-0.14*tc-0.12*dv-0.08*rare-0.06*shock-0.04*site_complex+trans_eff*0.18+rng.normal(0,0.025),0.50,0.95))
        mape = float(np.clip(1-acc+rng.normal(0,0.01),0.05,0.45))
        thresh = int((ts>0.60) or (shock==1 and acc>0.70) or (dv>0.74))
        team_ai = 1.00*state+0.42*acc+0.16*tc+0.10*dv+0.18*trans_eff+0.18*heter-0.16*shock+rng.normal(0,0.09)
        team_peer = 0.60*state+0.22*acc+0.15*heter+0.10*trans_eff-0.16*shock+rng.normal(0,0.09)
        team_prev = 0.46*state+0.28*prev_perf+0.18*acc+0.10*heter-0.18*shock+rng.normal(0,0.09)
        if p > 1:
            lp = -0.45+1.85*state+1.10*team_ai+0.70*team_peer+0.55*team_prev+0.24*thresh+0.24*trans_eff+0.30*heter-0.38*shock
            if state == 0:
                lp -= 0.25*shock
            else:
                lp += 0.28*thresh
            state = int(rng.random() < 1/(1+np.exp(-lp)))
        lat = float(np.clip(6.0+5.0*tc+4.0*dv+2.2*amb+1.6*shock-0.9*transparency-1.1*state-0.9*acc+rng.normal(0,0.35),1.5,30))
        ai_lin = -1.35+1.70*state+0.95*tc+0.32*dv+0.80*acc-0.52*ts-0.42*shock+0.70*trans_eff+0.24*heter+0.14*thresh+0.50*thresh*state+rng.normal(0,0.07)
        ai = 1/(1+np.exp(-ai_lin))
        esc_lin = -2.50+1.18*state+0.86*ts+1.25*amb+0.84*rare+0.76*shock+0.24*acc+0.14*tc+0.11*lat/10+0.10*thresh+0.42*thresh*state+0.12*trans_eff+0.22*orch+rng.normal(0,0.07)
        esc = 1/(1+np.exp(-esc_lin))
        total = ai + esc
        max_total = 0.90 if state == 1 else 0.55
        if total > max_total:
            scl = max_total / total
            ai *= scl; esc *= scl
        ov = max(0.04, 1-ai-esc)
        total2 = ai + esc + ov
        ai, esc, ov = ai/total2, esc/total2, ov/total2
        sld = 1.10-1.95*state-1.40*ai-1.15*esc+0.82*shock+0.42*dv+0.25*tc+rng.normal(0,0.12)
        icd = 0.95-1.45*state-0.95*ai-0.70*esc+0.46*shock+0.22*tc+0.16*ts+rng.normal(0,0.12)
        ecd = 0.90-1.35*state-0.70*ai-1.00*esc+0.78*shock+0.42*rare+0.28*dv+rng.normal(0,0.12)
        err = int(max(0, round(1.7-1.00*state-0.70*ai-1.00*esc+1.00*shock+0.50*rare+rng.normal(0,0.30))))
        prev_perf = (-(sld+icd+ecd)/3.0)-0.18*err
        rows.append({
            'manager_id': m['manager_id'], 'period_id': p,
            'ai_decision_authority_share': ai, 'override_rate': ov, 'escalation_rate': esc,
            'decision_latency_avg': lat, 'service_level_delta': sld, 'inventory_cost_delta': icd,
            'expedite_cost_delta': ecd, 'error_incident_count': err, 'target_difficulty': ts,
            'performance_pressure_index': float(np.clip(0.30+0.40*hp+0.15*ts+rng.normal(0,0.04),0,1)),
            'recent_negative_shock': shock, 'task_complexity_index': tc, 'demand_volatility': dv,
            'supply_disruption_count': int(shock+rare+(rng.random()<(0.08+0.16*dv))),
            'forecast_accuracy_mape': mape, 'transparency_level': transparency, 'ai_implementation_age': p,
            'ai_accuracy': acc, 'task_stakes': ts, 'threshold_trigger': thresh,
            'team_vs_tminus1': team_prev, 'team_vs_peer': team_peer, 'team_ai_vs_without_ai': team_ai,
            'rare_event': rare, 'ambiguity_index': amb, 'manager_heterogeneity': heter,
            'latent_state_v2': state, 'ai_version': 'v2'
        })
panel_manager = pd.DataFrame(rows)
site_lookup = manager_master.set_index('manager_id')['site_id'].to_dict()
rows = []; ep = 0
for _, r in panel_manager.iterrows():
    n_tasks = int(30 + 15*r.task_complexity_index + 8*r.demand_volatility + rng.integers(5,15))
    n_escalated = int(round(r.escalation_rate*n_tasks)); n_accepted = int(round(r.ai_decision_authority_share*n_tasks)); n_rejected = int(round(r.override_rate*0.45*n_tasks)); n_modified = max(0, n_tasks-n_escalated-n_accepted-n_rejected)
    n_modified += (n_tasks-(n_escalated+n_accepted+n_rejected+n_modified))
    actions = (['accept']*n_accepted)+(['modify']*n_modified)+(['reject']*n_rejected)
    rng.shuffle(actions)
    flags = np.array([1]*n_escalated+[0]*(n_tasks-n_escalated))
    rng.shuffle(flags)
    for i in range(n_tasks):
        ep += 1
        action = actions[i] if i < len(actions) else 'modify'
        ai_conf = float(np.clip(0.45+0.50*r.ai_accuracy-0.10*r.ambiguity_index-0.06*r.recent_negative_shock+rng.normal(0,0.04),0.05,0.99))
        rows.append({
            'episode_id': f'EP_{ep:07d}', 'manager_id': r.manager_id, 'period_id': int(r.period_id),
            'site_id': site_lookup[r.manager_id], 'ai_version': 'v2',
            'ai_recommendation_type': rng.choice(['transfer','reroute','reorder','expedite']),
            'ai_confidence': ai_conf, 'ai_uncertainty': 1-ai_conf,
            'explanation_provided': int(r.transparency_level>0), 'manager_action': action,
            'override_flag': int(action in ['modify','reject']), 'escalation_flag': int(flags[i]),
            'time_to_decision': float(np.clip(r.decision_latency_avg+rng.normal(0,0.9),0.5,40)),
        })
decision_episode = pd.DataFrame(rows)
orig['panel_manager_period'] = panel_manager
orig['decision_episode'] = decision_episode
orig['ai_system_master'] = pd.DataFrame([{
    'ai_version':'v2','deployment_date':'2017-01-01','autonomy_level':'high',
    'explanation_capability':'4-level schedule (0 none, 1 basic, 2 moderate, 3 detailed)',
    'confidence_calibration_score':0.90
}])
with pd.ExcelWriter(output_xlsx, engine='openpyxl') as writer:
    for name in xls.sheet_names:
        orig[name].to_excel(writer, index=False, sheet_name=name)
print(f'Saved {output_xlsx}')
