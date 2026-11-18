"""Eye-on-base calibration loading and optical-frame coordinate conversion."""
import numpy as np
import yaml

OPTICAL_TO_CAMERA = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])


def load_calibration(path, optical_input=True):
    with open(path, encoding='utf-8') as stream:
        doc = yaml.safe_load(stream)
    parameters, transform = doc['parameters'], doc['transform']
    if parameters['calibration_type'] != 'eye_on_base':
        raise ValueError('Only eye_on_base calibration is supported')
    source, target = parameters['tracking_base_frame'], parameters['robot_base_frame']
    if not source or not target or source == target:
        raise ValueError('Invalid calibration source/target frames')
    q = np.array([transform['rotation'][k] for k in ('x', 'y', 'z', 'w')], dtype=float)
    translation = np.array([transform['translation'][k] for k in ('x', 'y', 'z')], dtype=float)
    if not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12 or not np.isfinite(translation).all():
        raise ValueError('Invalid calibration quaternion/translation')
    x, y, z, w = q / np.linalg.norm(q)
    rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                         [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                         [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    source_optical = source.endswith('_optical_frame')
    if optical_input and not source_optical:
        rotation = rotation @ OPTICAL_TO_CAMERA
    elif not optical_input and source_optical:
        rotation = rotation @ OPTICAL_TO_CAMERA.T
    return rotation, translation, source, target


def transform_xyz_buffer(data, width, height, point_step, row_step, offsets,
                         bigendian, rotation, translation):
    if width < 0 or height < 0 or point_step <= 0 or row_step < width*point_step or len(data) != height*row_step:
        raise ValueError('Malformed PointCloud2 layout')
    if len(set(offsets)) != 3 or any(o < 0 or o+4 > point_step for o in offsets):
        raise ValueError('Invalid XYZ offsets')
    output = bytearray(data)
    if width == 0 or height == 0:
        return output
    dtype = np.dtype({'names': ['x', 'y', 'z'], 'formats': [('>' if bigendian else '<')+'f4']*3,
                      'offsets': offsets, 'itemsize': point_step})
    view = np.ndarray((height, width), dtype=dtype, buffer=output, strides=(row_step, point_step))
    xyz = np.stack([view[k] for k in ('x', 'y', 'z')], axis=-1).astype(np.float64)
    valid = np.isfinite(xyz).all(axis=-1)
    xyz[valid] = xyz[valid] @ rotation.T + translation
    for i, key in enumerate(('x', 'y', 'z')):
        view[key][valid] = xyz[..., i][valid]
    return output


def rotation_quaternion(rotation):
    """Return normalized xyzw using the symmetric rotation-matrix eigenproblem."""
    r = rotation
    k = np.array([
        [r[0,0]-r[1,1]-r[2,2], r[0,1]+r[1,0], r[0,2]+r[2,0], r[2,1]-r[1,2]],
        [r[0,1]+r[1,0], r[1,1]-r[0,0]-r[2,2], r[1,2]+r[2,1], r[0,2]-r[2,0]],
        [r[0,2]+r[2,0], r[1,2]+r[2,1], r[2,2]-r[0,0]-r[1,1], r[1,0]-r[0,1]],
        [r[2,1]-r[1,2], r[0,2]-r[2,0], r[1,0]-r[0,1], np.trace(r)]])
    _, vectors = np.linalg.eigh(k)
    q = vectors[:, -1]
    return q if q[3] >= 0 else -q


def invert_rigid_transform(rotation, translation):
    """Invert parent<-child into child<-parent; translation is in parent axes."""
    rotation = np.asarray(rotation, dtype=float)
    translation = np.asarray(translation, dtype=float)
    if (rotation.shape != (3, 3) or translation.shape != (3,)
            or not np.isfinite(rotation).all() or not np.isfinite(translation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10)
            or not np.isclose(np.linalg.det(rotation), 1.0)):
        raise ValueError('Expected a finite rigid transform with proper rotation')
    return rotation.T, -rotation.T @ translation


def camera_to_left_transform(rotation_b_c, translation_b_c, left_origin_in_right):
    """A<-C = inverse(B<-A) * B<-C; A has opposite X/Y and common Z."""
    rotation_b_c = np.asarray(rotation_b_c, dtype=float)
    translation_b_c = np.asarray(translation_b_c, dtype=float)
    # Validate calibration transform as well as the inter-base transform.
    invert_rigid_transform(rotation_b_c, translation_b_c)
    rotation_a_b, translation_a_b = invert_rigid_transform(
        np.diag([-1., -1., 1.]), left_origin_in_right)
    return rotation_a_b @ rotation_b_c, rotation_a_b @ translation_b_c + translation_a_b


def camera_obstacle_pose(path, center_optical):
    """Box pose in B, using exactly the optical transform used by cable clouds."""
    offset = np.asarray(center_optical, dtype=float)
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError('Camera obstacle center must contain three finite optical coordinates')
    rotation, translation, _, target = load_calibration(path, optical_input=True)
    if target != 'fr3_link0':
        raise ValueError('Camera obstacle calibration must target fr3_link0')
    return [float(v) for v in np.r_[rotation @ offset + translation, rotation_quaternion(rotation)]]
