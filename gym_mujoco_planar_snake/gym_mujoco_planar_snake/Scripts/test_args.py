import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--list1', nargs='*', type=int)

parser.add_argument('--list2', nargs='*', type=int)

args = parser.parse_args()

print(args.list1)
print(args.list2)
print("Success")