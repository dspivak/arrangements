"""Reproduce P29's measured summaries; no new parameter search.

Run from the repository: timeout 60 .venv/bin/python scripts/town_report.py.
Streaming one stage at a time; whole job expected below 10 seconds.
"""
from dataclasses import replace
import json
from pathlib import Path
import sys
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arrangements.town import Settings, build, run, summary


def main():
    c = Settings()
    scenarios = [('no_return', build(1, c)), ('law_prompt', build(2, replace(c, delay_steps=0))),
                 ('law_delayed', build(2, c)), ('law_informed', build(2, c, informed=True)),
                 ('insurance_past', build(3, c)), ('insurance_account', build(4, c)),
                 ('insurance_past_half_step', build(3, replace(c, dt=.01, delay_steps=600))),
                 ('insurance_account_half_step', build(4, replace(c, dt=.01, delay_steps=600)))]
    results = {}
    for name, town in scenarios:
        trajectory = run(town)
        result = summary(town, trajectory)
        output = town.value(trajectory, 'output')[:, 0]
        result['operating_surplus'] = float(town.settings.dt*np.sum(
            town.settings.benefit*output[:-1]-output[:-1]**2/2))
        if 'money' in town.slots:
            money = town.value(trajectory, 'money')
            paid = np.flatnonzero(np.diff(money[:, 2]) > 1e-12)
            result['first_claim_paid_at'] = float((paid[0]+1)*town.settings.dt) if len(paid) else None
            result['final_treasury'] = float(money[-1, 0])
            result['final_compensation'] = float(money[-1, 2])
            result['unpaid_incurred_claims'] = result['harm']-float(money[-1, 2])
            if town.stage >= 3:
                result['fund_less_unpaid_claims'] = float(money[-1, 1])-result['unpaid_incurred_claims']
        results[name] = result
        print(name+': '+json.dumps(result, sort_keys=True), flush=True)
    destination = Path(__file__).resolve().parents[1]/'notes'/'P29-TOWN-MEASUREMENTS.json'
    destination.write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__':
    main()
