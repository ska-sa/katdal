#
# Extract sensor data for use in katdal unit tests.
#
# Ludwig Schwardt
# 2021-11-09
#

import argparse
import copy

import katpoint

import katdal
from katdal.categorical import tabulate_categorical


def letter_labels(seq):
    ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    assert len(seq) <= 26, 'Make the alphabet bigger!'
    return list(ALPHABET[:len(seq)])


description = 'Extract raw and aligned sensors relevant to scans.'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('dataset')
parser.add_argument('--dumps', type=int, default=0, metavar='N',
                    help='Only extract the first N dumps')
args = parser.parse_args()

kwargs = {}
if args.dumps:
    kwargs['preselect'] = dict(dumps=slice(0, args.dumps))
d = katdal.open(args.dataset, **kwargs)
names = 'Label Target Scan'.split()
nothing = katpoint.Target('Nothing, special')

label_in = d.sensor.get('obs_label')
target_in = d.sensor.get(f'Antennas/{d.ref_ant}/target')
target_in = copy.deepcopy(target_in)
targets = target_in.unique_values
lookup = dict(zip(targets, letter_labels(targets)))
# Preserve any "Nothing, special" target since alignment needs it
lookup[nothing] = 'Nothing, special'
target_in.unique_values = [lookup[t] for t in targets]
scan_in = d.sensor.get(f'Antennas/{d.ref_ant}/activity')
sensors_in = dict(zip(names, [label_in, target_in, scan_in]))
print(tabulate_categorical(sensors_in))

label_out = d.sensor.get('Observation/label')
target_out = d.sensor.get('Observation/target')
target_out = copy.deepcopy(target_out)
targets = target_out.unique_values
target_out.unique_values = [lookup[t] for t in targets]
scan_out = d.sensor.get('Observation/scan_state')
sensors_out = dict(zip(names, [label_out, target_out, scan_out]))
print(tabulate_categorical(sensors_out))
