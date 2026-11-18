"""Strict source ownership and freshness for measured dual-arm joint states."""
import math


class MeasuredJointStore:
    def __init__(self, stale_timeout=1.0, required_arms=("a", "b")):
        if not math.isfinite(stale_timeout) or stale_timeout <= 0:
            raise ValueError('stale_timeout must be positive')
        if not required_arms or not set(required_arms).issubset({"a", "b"}):
            raise ValueError("required_arms must contain a and/or b")
        self.stale_timeout = stale_timeout
        self.owners = {}
        self.required = []
        self.order = []
        for arm, prefix in [('b', ''), ('a', 'a_')]:
            joints = [prefix + 'fr3_joint' + str(i) for i in range(1, 8)]
            fingers = [prefix + 'fr3_finger_joint1', prefix + 'fr3_finger_joint2']
            self.owners[arm] = set(joints)
            self.owners[arm + '_gripper'] = set(fingers)
            if arm not in required_arms:
                continue
            self.required += joints
            self.order += joints + fingers
        self.values = {}

    def update(self, source, names, positions, velocities, efforts, stamp_ns, received):
        allowed = self.owners[source]
        if len(names) != len(positions) or len(names) != len(set(names)):
            raise ValueError('Joint names/positions mismatch or duplicate names')
        if any(values and len(values) != len(names) for values in (velocities, efforts)):
            raise ValueError('Malformed velocity/effort array')
        if stamp_ns <= 0:
            raise ValueError('Missing measured timestamp')
        unexpected = []
        # Validate all owned data before accepting any of it.
        for i, name in enumerate(names):
            if name in allowed and not math.isfinite(positions[i]):
                raise ValueError('Nonfinite measured joint position: ' + name)
        for i, name in enumerate(names):
            if name not in allowed:
                unexpected.append(name)
                continue
            if name in self.values and stamp_ns < self.values[name][3]:
                continue
            velocity = velocities[i] if velocities and math.isfinite(velocities[i]) else None
            effort = efforts[i] if efforts and math.isfinite(efforts[i]) else None
            self.values[name] = (positions[i], velocity, effort, stamp_ns, received)
        return unexpected

    def fresh(self, name, now_ns, received_now):
        if name not in self.values:
            return False
        value = self.values[name]
        age = (now_ns - value[3]) * 1e-9
        return (-.5 <= age <= self.stale_timeout
                and 0 <= received_now - value[4] <= self.stale_timeout)

    def snapshot(self, now_ns, received_now):
        if not all(self.fresh(name, now_ns, received_now) for name in self.required):
            return None
        names = [name for name in self.order if self.fresh(name, now_ns, received_now)]
        positions = [self.values[name][0] for name in names]
        velocities = [self.values[name][1] for name in names]
        efforts = [self.values[name][2] for name in names]
        if any(value is None for value in velocities):
            velocities = []
        if any(value is None for value in efforts):
            efforts = []
        # Never re-stamp old hardware data as a fresh measurement.
        stamp_ns = min(self.values[name][3] for name in self.required)
        return names, positions, velocities, efforts, stamp_ns
