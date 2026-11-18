"""Build combined planning model from the installed, matching FR3 descriptions."""
import copy
import os
import xml.etree.ElementTree as ET
import xacro
from ament_index_python.packages import get_package_share_directory


def build_model():
    description = get_package_share_directory('franka_description')
    model = ET.Element('robot', name='cable_dual_fr3')
    semantic = ET.Element('robot', name='cable_dual_fr3')
    for prefix in ('', 'a'):
        mappings = dict(arm_prefix=prefix, hand='true', ros2_control='false')
        urdf = ET.fromstring(xacro.process_file(os.path.join(description, 'robots/fr3/fr3.urdf.xacro'), mappings=mappings).toxml())
        srdf = ET.fromstring(xacro.process_file(os.path.join(description, 'robots/fr3/fr3.srdf.xacro'), mappings=mappings).toxml())
        for element in urdf:
            if prefix == 'a':
                if element.tag == 'link' and element.get('name') == 'base':
                    continue
                if element.tag == 'joint' and element.find('child').get('link') == 'a_fr3_link0':
                    element.find('parent').set('link', 'fr3_link0')
                    origin = element.find('origin')
                    origin.set('xyz', '0.47 0 0')
                    origin.set('rpy', '0 0 3.141592653589793')
            model.append(copy.deepcopy(element))
        for element in srdf:
            if element.tag != 'virtual_joint':
                semantic.append(copy.deepcopy(element))
    # Retain vendor group names for direct reuse of per-arm kinematics configs.
    combined = ET.SubElement(semantic, 'group', name='both_arms')
    ET.SubElement(combined, 'group', name='fr3_arm')
    ET.SubElement(combined, 'group', name='a_fr3_arm')
    # No inter-arm collision exemptions are added.
    return {'robot_description': ET.tostring(model, encoding='unicode'),
            'robot_description_semantic': ET.tostring(semantic, encoding='unicode')}
