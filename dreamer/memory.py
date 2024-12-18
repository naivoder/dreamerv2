import collections
import datetime
import pathlib
import io
import uuid
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader


class ReplayBuffer:
    def __init__(
        self,
        save_dir="logs",
        capacity=1000,
        batch_size=50,
        sequence_length=50,
        min_episode_length=1,
        max_episode_length=0,
        num_workers=2,
        prefetch_factor=5,
        sample_ongoing=False,
        prioritize_ends=False,
        verbose=False,
    ):
        self.save_dir = pathlib.Path(save_dir).expanduser()
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.sample_ongoing = sample_ongoing
        self.prioritize_ends = prioritize_ends

        self.capacity = capacity
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.min_episode_length = min_episode_length
        self.max_episode_length = max_episode_length

        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

        self.ongoing = collections.defaultdict(lambda: collections.defaultdict(list))
        self.completed = collections.OrderedDict(
            load_episodes(self.save_dir, capacity, min_episode_length)
        )

        self.loaded_episodes = len(self.completed)
        self.loaded_steps = sum(episode_length(x) for x in self.completed.values())
        self.total_episodes, self.total_steps = count_episodes(save_dir)

        self.verbose = verbose

    @property
    def stats(self):
        return {
            "total_episodes": self.total_episodes,
            "total_steps": self.total_steps,
            "loaded_episodes": self.loaded_episodes,
            "loaded_steps": self.loaded_steps,
        }

    def add_step(self, transition, worker=0):
        episode = self.ongoing[worker]

        for key, value in transition.items():
            episode[key].append(value)

        if transition["done"]:
            self.add_episode(episode)
            episode.clear()

    def add_episode(self, episode):
        length = episode_length(episode)

        if length < self.min_episode_length:
            if self.verbose:
                print(f"Episode too short ({length}), skipping...")
            return

        self.total_steps += length
        self.loaded_steps += length
        self.total_episodes += 1
        self.loaded_episodes += 1

        episode = {k: cast_dtype(v) for k, v in episode.items()}
        filename = save_episode(self.save_dir, episode)
        self.completed[str(filename)] = episode

        self._enforce_capacity()

    def sample_experiences(self):
        dataset = ExperienceDataset(self._generate_chunks)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def _generate_chunks(self):
        sequence = self._sample_sequence()

        while True:
            chunks = collections.defaultdict(list)

            total_steps = 0
            while total_steps < self.sequence_length:
                n_steps = self.sequence_length - total_steps
                chunk = {k: v[:n_steps] for k, v in sequence.items()}
                sequence = {k: v[n_steps:] for k, v in sequence.items()}

                for key, value in chunk.items():
                    chunks[key].append(value)

                total_steps += len(chunk["action"])

                if len(sequence["action"]) == 0:
                    sequence = self._sample_sequence()

            chunks = {k: np.concatenate(v) for k, v in chunks.items()}
            yield chunks

    def _sample_sequence(self):
        episodes = list(self.completed.values())

        if self.sample_ongoing:
            episodes.extend(
                x
                for x in self.ongoing.values()
                if episode_length(x) > self.min_episode_length
            )

        episode = np.random.choice(episodes)
        length = total = len(episode["action"])

        if self.max_episode_length:
            length = min(length, self.max_episode_length)

        # Randomize length to avoid all chunks completing at the same time
        length -= np.random.randint(self.min_episode_length)
        length = max(self.min_episode_length, length)
        upper = total - length + 1

        if self.prioritize_ends:
            upper += self.min_episode_length

        idx = min(np.random.randint(upper), total - length)
        sequence = {
            k: cast_dtype(v[idx : idx + length])
            for k, v in episode.items()
            if not k.startswith("log_")
        }
        sequence["start"] = np.zeros(len(sequence["action"]), dtype=np.bool)
        sequence["start"][0] = True

        if self.max_episode_length:
            assert len(sequence["action"]) <= self.max_episode_length

        return sequence

    def _enforce_capacity(self):
        while self.loaded_episodes > 1 and self.loaded_steps > self.capacity:
            episode = self.completed.popitem(last=False)[1]
            self.loaded_steps -= episode_length(episode)
            self.loaded_episodes -= 1


class ExperienceDataset(IterableDataset):
    def __init__(self, generator):
        self.generator = generator

    def __iter__(self):
        for chunk in self.generator():
            yield {k: torch.tensor(v) for k, v in chunk.items()}


def collate_fn(batch):
    keys = batch[0].keys()
    return {k: torch.stack([d[k] for d in batch], dim=0) for k in keys}


def save_episode(save_dir, episode):
    timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    run_id = str(uuid.uuid4().hex)
    length = episode_length(episode)
    filename = save_dir / f"{timestamp}-{run_id}-{length}.npz"

    with io.BytesIO() as buffer:
        np.savez_compressed(buffer, **episode)
        buffer.seek(0)

        with filename.open("wb") as f:
            f.write(buffer.read())

    return filename


def load_episodes(save_dir, capacity=None, min_len=1):
    filenames = sorted(save_dir.glob("*.npz"))
    episodes = {}

    if capacity:
        num_steps = 0
        num_episodes = 0

        for filename in filenames:
            num_steps += int(str(filename).split("-")[-1][:-4])
            num_episodes += 1

            if num_steps >= capacity:
                break

        filenames = filenames[-num_episodes:]

    for filename in filenames:
        try:
            with filename.open("rb") as f:
                episode = np.load(f)
                episode = {k: episode[k] for k in episode.keys()}

        except Exception as e:
            print(f"Error loading episode {str(filename)}: {e}")
            continue

        episodes[str(filename)] = episode

    return episodes


def count_episodes(save_dir):
    filenames = list(save_dir.glob("*.npz"))
    num_episodes = len(filenames)
    num_steps = sum(int(str(n).split("-")[-1][:-4]) for n in filenames)
    return num_episodes, num_steps


def episode_length(episode):
    return len(episode["action"]) - 1


def cast_dtype(value):
    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
        return value.astype(np.float32)
    elif np.issubdtype(value.dtype, np.signedinteger):
        return value.astype(np.int32)
    elif np.issubdtype(value.dtype, np.uint8):
        return value.astype(np.uint8)
    return value
