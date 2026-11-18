"""Validate robot capability descriptions against the existing deterministic executor."""
import json
from pathlib import Path


def load_skill_task(catalog_path, task_path, states, commands, transitions):
    catalog_path, task_path = Path(catalog_path), Path(task_path)
    catalog = json.loads(catalog_path.read_text(encoding='utf-8'))
    task = json.loads(task_path.read_text(encoding='utf-8'))
    if catalog.get('schema_version') != 1 or task.get('schema_version') != 1:
        raise ValueError('Unsupported robot skill/task schema')
    skills = {}
    for filename in catalog['skills']:
        path = (catalog_path.parent / filename).resolve()
        if path.parent != catalog_path.parent.resolve():
            raise ValueError('Skill descriptor must reside beside catalog')
        skill = json.loads(path.read_text(encoding='utf-8'))
        name = skill['name']
        if name in skills or skill.get('schema_version') != 1:
            raise ValueError('Duplicate skill or unsupported skill schema')
        if skill['parameters']['properties']['arm']['enum'] != ['a', 'b']:
            raise ValueError('Invalid skill arm contract')
        if not skill.get('preconditions') or not skill.get('success_conditions'):
            raise ValueError('Skill needs preconditions and success criteria')
        skills[name] = skill
    if task.get('executor') != 'alternate_grasp_state_machine':
        raise ValueError('Task requires a different executor; arbitrary JSON transitions are not executed')
    rows = task['states']
    if len(rows) != len(states) or len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Task state set differs from implemented state machine')
    by_id = {row['id']: row for row in rows}
    for state in states:
        row = by_id[state.value]
        if row.get('command') != commands.get(state) or row['next'] != transitions[state].value:
            raise ValueError('Task description differs from implemented transition: ' + state.value)
        command = commands.get(state)
        if command and command.rsplit('_', 1)[0] not in skills:
            raise ValueError('Task references an unavailable skill: ' + command)
    return skills, task
