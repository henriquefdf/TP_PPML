
import pandas as pd
import numpy as np

def partition_train_data(
    train_path="./processed/train.parquet",
    output_dir="./partitions",
    num_clients=10,
    random_state=42,
):
    # 1) Load your stratified train split
    train_df = pd.read_parquet(train_path)

    # 2) Shuffle then split into roughly equal shards
    shuffled = train_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    shards = np.array_split(shuffled, num_clients)

    # 3) Write each shard to disk
    for i, shard in enumerate(shards, start=1):
        shard.to_parquet(f"{output_dir}/client_{i:02d}.parquet", index=False)
        print(f"Wrote shard {i:02d} with {len(shard)} examples")

if __name__ == "__main__":
    partition_train_data()
