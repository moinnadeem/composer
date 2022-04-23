from composer.datasets import c4
from tqdm import tqdm

# import tracemalloc
# tracemalloc.start()

c4_dataset = c4.C4Dataset(split="train",
                          num_samples=275184000,
                          tokenizer_name="bert-base-uncased",
                          max_seq_len=128,
                          group_method="truncate")

c4_dataset = iter(c4_dataset)

for _ in tqdm(range(int(1e7))):
    next(c4_dataset)

# snapshot = tracemalloc.take_snapshot()
# top_stats = snapshot.statistics('lineno')

# print("[ Top 10 ]")
# for stat in top_stats[:10]:
    # print(stat)
