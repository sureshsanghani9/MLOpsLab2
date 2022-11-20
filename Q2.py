import argparse

import os
import sys

parser = argparse.ArgumentParser(description='clf_name and random_state', prefix_chars='--')

# Add the arguments
parser.add_argument('clf_name', metavar='clf_name', type=str, help='clf_name')
parser.add_argument('random_state', metavar='random_state', type=int, help='random_state')

args = parser.parse_args()

clf_name = args.clf_name
random_state = args.random_state

print(clf_name)
print(random_state)