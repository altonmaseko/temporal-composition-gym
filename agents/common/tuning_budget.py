"""Conservative fixed role allowances; elapsed rental time includes idle time."""
from datetime import datetime, timezone
import math

ROLE_HOURS = {'a': 33.34, 'b': 33.33, 'c': 33.33}
REFERENCE = {('delivery','dqn'):.16,('delivery','dqn_ts'):.16,('delivery','drqn'):.51,('delivery','qrm'):.11,
             ('colour','dqn'):.98,('colour','dqn_ts'):1.08,('colour','drqn'):3.66,('colour','qrm_cue'):.72,
             ('maintenance','ppo'):.42,('maintenance','recurrent_ppo'):1.,('maintenance','ppo_rm'):.32}


def remaining_seconds(role, billing_start, previously_billed=0., hourly_rate=.30, now=None):
    start=datetime.fromisoformat(billing_start.replace('Z','+00:00'))
    if (start.tzinfo is None or not math.isfinite(previously_billed) or not math.isfinite(hourly_rate)
            or previously_billed < 0 or hourly_rate <= 0):
        raise ValueError('billing start must include timezone; billed hours nonnegative and rate positive')
    now=now or datetime.now(timezone.utc)
    if start > now:
        raise ValueError('billing start cannot be in the future')
    # Split both caps across roles; previous machines/attempts must be carried forward.
    allowance=min(ROLE_HOURS[role], ROLE_HOURS[role] * .30 / hourly_rate)
    return (allowance-previously_billed)*3600-(now-start).total_seconds()


def projected_seconds(env,algo,observed=()):
    return max([REFERENCE[env,algo]*.25*3*3600, *[v*1.25 for v in observed]])
