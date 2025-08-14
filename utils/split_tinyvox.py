import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser(
        description='Split TinyVox into a training, validation, and test sets',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data', required=True,
                        help='Path to the TinyVox folder containing metadata.csv.')
    parser.add_argument('--val_prop', type=float, default=0.05)
    parser.add_argument('--test_prop', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    args.data = Path(args.data)

    metadata = pd.read_csv(args.data / 'metadata.csv')

    np.random.seed(args.seed)
    # Filter empty rows
    metadata = metadata[metadata['phones'] != ' |']

    # Calculate target row counts
    total_rows = len(metadata)
    test_rows_target = int(total_rows * args.test_prop)
    val_rows_target = int(total_rows * args.val_prop)

    # Get children and their row counts
    child_counts = metadata.groupby('child_pseudoid').size().sort_values(ascending=False)
    children = child_counts.index.tolist()
    np.random.shuffle(children)

    # Greedily assign children to splits to match target proportions
    test_children = []
    val_children = []
    test_rows_current = 0
    val_rows_current = 0

    for child in children:
        child_row_count = child_counts[child]

        # Assign to test if we haven't reached target
        if test_rows_current < test_rows_target:
            test_children.append(child)
            test_rows_current += child_row_count
        # Assign to validation if we haven't reached target
        elif val_rows_current < val_rows_target:
            val_children.append(child)
            val_rows_current += child_row_count
        # Otherwise assign to training
        else:
            break

    # Remaining children go to training
    train_children = [child for child in children
                      if child not in test_children and child not in val_children]

    # Create the splits
    train_df = metadata[metadata['child_pseudoid'].isin(train_children)].copy()
    val_df = metadata[metadata['child_pseudoid'].isin(val_children)].copy()
    test_df = metadata[metadata['child_pseudoid'].isin(test_children)].copy()

    # Compute durations
    total_duration = (metadata['offset'] - metadata['onset']).sum() / 3600000
    train_duration = (train_df['offset'] - train_df['onset']).sum() / 3600000
    val_duration = (val_df['offset'] - val_df['onset']).sum() / 3600000
    test_duration = (test_df['offset'] - test_df['onset']).sum() / 3600000

    # Count unique children
    total_children = metadata['child_pseudoid'].nunique()
    train_children_count = len(train_children)
    val_children_count = len(val_children)
    test_children_count = len(test_children)

    # Print actual proportions achieved
    print(f"Target proportions - Val: {args.val_prop:.1%}, Test: {args.test_prop:.1%}")
    print(f"Actual proportions - Val: {len(val_df) / total_rows:.1%}, Test: {len(test_df) / total_rows:.1%}")
    print(f"Rows - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(f"Children - Train: {train_children_count}, Val: {val_children_count}, Test: {test_children_count}, Total: {total_children}")

    print(f"Duration (hours) - Train: {train_duration:.1f}, Val: {val_duration:.1f}, Test: {test_duration:.1f}")
    print(f"Duration proportions - Val: {val_duration / total_duration:.1%}, Test: {test_duration / total_duration:.1%}")
    print(f"Total duration: {total_duration:.1f} hours")

    for name, fold in zip(['train', 'val', 'test'], [train_df, val_df, test_df]):
        fold.to_csv(args.data / f'{name}.csv', index=False)

if __name__ == "__main__":
    main()
