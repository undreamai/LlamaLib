import os
import yaml
import re

yaml.SafeDumper.org_represent_str = yaml.SafeDumper.represent_str
yaml.width = 2 ** 32


def repr_str(dumper, data):
    if '\n' in data:
        return dumper.represent_scalar(u'tag:yaml.org,2002:str', data, style='|')
    return dumper.org_represent_str(data)

yaml.add_representer(str, repr_str, Dumper=yaml.SafeDumper)

def load_yaml(filename):
    with open(filename, 'r') as file:
        return yaml.safe_load(file)

def extract_steps(yaml_content):
    jobs = yaml_content.get('jobs', {})
    steps_dict = {}

    for job, job_content in jobs.items():
        steps = job_content.get('steps', [])
        for step in steps:
            step_id = step.get('id')
            if step_id:
                steps_dict[step_id] = yaml.safe_dump([step]).strip()

    return steps_dict

def replace_placeholders(template_content, steps_dict):
    def replace_match(match):
        key = match.group(1)
        return steps_dict.get(key, f'@@{key}@@')

    pattern = r'@@(.*?)@@'
    lines = []
    for line in template_content.split('\n'):
        indices = [(m.start(0), m.end(0)) for m in re.finditer(pattern, line)]
        if indices:
            assert len(indices) == 1
            space = ''.join([' '] * indices[0][0])
            line = re.sub(pattern, replace_match, line).replace('\n', '\n' + space) + "\n"
        lines.append(line)
    return '\n'.join(lines)

def main(yaml_file, template_file, output_file):
    # Load the YAML content
    yaml_content = load_yaml(yaml_file)
    # Extract steps by their IDs
    steps_dict = extract_steps(yaml_content)

    # Read the template file
    with open(template_file, 'r') as file:
        template_content = file.read()

    # Replace placeholders in the template with steps from the YAML
    result_content = replace_placeholders(template_content, steps_dict)

    # Write the result to the output file
    with open(output_file, 'w') as file:
        file.write(result_content)

    print(f"Output written to {output_file}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_file = os.path.join(script_dir, 'build_library_steps.yaml')
    template_file = os.path.join(script_dir, 'build_library_template.yaml')
    output_file = os.path.join(script_dir, '..', 'workflows', 'build_library.yaml')
    main(yaml_file, template_file, output_file)
