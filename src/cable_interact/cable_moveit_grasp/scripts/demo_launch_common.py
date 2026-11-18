"""Shared demo launch helpers. No robot connections are made here."""
import atexit
import fcntl
import hashlib
import os

_LOCKS = {}


def acquire_robot_leases(robot_ips):
    """Keep OS locks for this launch process; a crash releases them automatically.

    Covers these two demo launches, not external ROS action clients or vendor tools.
    """
    acquired = []
    try:
        for ip in sorted(set(robot_ips)):
            key = hashlib.sha256(ip.encode()).hexdigest()[:20]
            if key in _LOCKS:
                raise RuntimeError('Robot already claimed by another demo in this launch: ' + ip)
            fd = os.open('/tmp/cable_moveit_demo_' + key + '.lock', os.O_CREAT | os.O_RDWR, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(fd)
                raise RuntimeError('Robot ' + ip + ' is already reserved by another cable MoveIt demo; stop that launch first')
            acquired.append((key, fd))
        _LOCKS.update(acquired)
    except Exception:
        for _, fd in acquired:
            os.close(fd)
        raise


def close_robot_leases():
    for fd in _LOCKS.values():
        os.close(fd)
    _LOCKS.clear()


atexit.register(close_robot_leases)
