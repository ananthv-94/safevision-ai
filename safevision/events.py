import csv
from dataclasses import dataclass, field
from pathlib import Path
import cv2
import numpy as np

DEFAULT_ZONE = [[0.32, 0.38], [0.82, 0.38], [0.95, 0.96], [0.20, 0.96]]
FIELDS = ['video_seconds', 'track_id', 'violation_type', 'zone_id']

def polygon(points, width, height):
    a = np.asarray(points, dtype=float)
    if a.ndim != 2 or a.shape[1] != 2 or len(a) < 3 or not np.isfinite(a).all():
        raise ValueError('Zone needs at least three finite [x,y] points.')
    if np.any(a < 0) or np.any(a > 1):
        raise ValueError('Zone coordinates must be between 0 and 1.')
    p = np.rint(a * [width - 1, height - 1]).astype(np.int32)
    # A convex polygon avoids ambiguous crossing edges in manually entered zones.
    if not cv2.isContourConvex(p) or cv2.contourArea(p) < 1:
        raise ValueError('Use a convex polygon with points ordered around its edge.')
    return p

def inside(box, zone):
    x1, _, x2, y2 = map(float, box)
    return cv2.pointPolygonTest(zone, ((x1+x2)/2, y2), False) >= 0

@dataclass
class Incidents:
    cooldown: float = 5.0
    exit_grace: float = 0.5
    rows: list = field(default_factory=list)
    states: dict = field(default_factory=dict)

    def update(self, track_id, is_inside, seconds):
        s = self.states.setdefault(track_id, {'active': False, 'outside_since': None, 'last': -float('inf')})
        if is_inside:
            s['outside_since'] = None
            if not s['active']:
                s['active'] = True
                if seconds-s['last'] >= self.cooldown:
                    self.rows.append(dict(video_seconds=round(seconds, 3), track_id=int(track_id),
                                          violation_type='restricted_zone_entry', zone_id='zone_1'))
                    s['last'] = seconds
                    return True
        elif s['active']:
            if s['outside_since'] is None:
                s['outside_since'] = seconds
            if seconds-s['outside_since'] >= self.exit_grace:
                s['active'] = False
        return False

    def save(self, path):
        with Path(path).open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(self.rows)
