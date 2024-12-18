import unittest
import tempfile
import shutil
import numpy as np
import torch
from pathlib import Path

from dreamer.memory import (
    ReplayBuffer,
    ExperienceDataset,
    collate_fn,
    save_episode,
    load_episodes,
    count_episodes,
    episode_length,
    cast_dtype,
)


class TestReplayBuffer(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for saving episodes
        self.temp_dir = tempfile.mkdtemp()
        self.save_path = Path(self.temp_dir)
        self.buffer = ReplayBuffer(
            save_dir=self.save_path, capacity=1000, batch_size=50, sequence_length=50
        )

    def tearDown(self):
        # Remove temporary directory and all created files
        shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        # Check that no episodes are loaded initially when directory is empty
        self.assertEqual(self.buffer.loaded_episodes, 0)
        self.assertEqual(self.buffer.loaded_steps, 0)
        self.assertEqual(self.buffer.total_episodes, 0)
        self.assertEqual(self.buffer.total_steps, 0)

    def test_episode_length_function(self):
        # Episode length is len(actions)-1
        episode = {"action": np.array([1, 2, 3, 4])}  # length should be 3
        self.assertEqual(episode_length(episode), 3)

    def test_cast_dtype_function(self):
        # Check various dtypes
        float_val = cast_dtype([1.2, 3.4])
        self.assertTrue(float_val.dtype == np.float32)
        int_val = cast_dtype([1, 2, 3])
        self.assertTrue(int_val.dtype == np.int32)

    def test_saving_and_loading_episodes(self):
        # Create a dummy episode
        episode = {
            "action": np.array([0, 1, 2, 3, 4], dtype=np.int32),
            "reward": np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32),
        }
        filename = save_episode(self.save_path, episode)

        # Ensure the file was created
        self.assertTrue(filename.exists())

        # Test loading
        loaded = load_episodes(self.save_path)
        self.assertEqual(len(loaded), 1)
        loaded_episode = list(loaded.values())[0]
        np.testing.assert_array_equal(loaded_episode["action"], episode["action"])
        np.testing.assert_array_equal(loaded_episode["reward"], episode["reward"])

    def test_count_episodes(self):
        # Save two episodes and count them
        ep1 = {"action": np.array([0, 1, 2], dtype=np.int32)}
        ep2 = {"action": np.array([0, 1, 2, 3], dtype=np.int32)}
        save_episode(self.save_path, ep1)
        save_episode(self.save_path, ep2)

        num_episodes, num_steps = count_episodes(self.save_path)
        self.assertEqual(num_episodes, 2)
        # steps = (len(action)-1) for each episode
        # ep1 steps = 2, ep2 steps = 3, total = 5
        self.assertEqual(num_steps, 5)

    def test_add_step_and_episode(self):
        # Add steps and complete an episode
        self.buffer.add_step({"action": 0, "reward": 1.0, "done": False})
        self.buffer.add_step({"action": 1, "reward": 1.0, "done": True})

        # After done: an episode should have been saved
        self.assertEqual(self.buffer.total_episodes, 1)
        self.assertEqual(self.buffer.loaded_episodes, 1)

        # Check that the episode is saved
        self.assertEqual(len(self.buffer.completed), 1)
        episode = list(self.buffer.completed.values())[0]
        np.testing.assert_array_equal(
            episode["action"], np.array([0, 1], dtype=np.int32)
        )

    def test_min_episode_length(self):
        # Set min_episode_length so short episodes are skipped
        self.buffer.min_episode_length = 3

        self.buffer.add_step({"action": 0, "reward": 1.0, "done": False})
        self.buffer.add_step({"action": 1, "reward": 1.0, "done": True})

        # Episode should be skipped
        self.assertEqual(self.buffer.loaded_episodes, 0)

    def test_capacity_enforcement(self):
        # Reduce capacity to a small number and add multiple episodes
        self.buffer.capacity = 10

        # Each episode will have 5 steps (action length=5), meaning 4 actual steps count
        for _ in range(5):
            self.buffer.add_step({"action": 0, "reward": 1.0, "done": False})
            self.buffer.add_step({"action": 1, "reward": 1.0, "done": False})
            self.buffer.add_step({"action": 2, "reward": 1.0, "done": False})
            self.buffer.add_step({"action": 3, "reward": 1.0, "done": False})
            self.buffer.add_step({"action": 4, "reward": 1.0, "done": True})

        # With capacity 10 steps max, after adding 5 episodes of 4 steps each,
        # we should have enforced capacity and removed older episodes.
        # We can't know exact episodes remain, but loaded steps should not exceed capacity.
        self.assertLessEqual(self.buffer.loaded_steps, 10)

    def test_sample_experiences(self):
        # Add a few episodes so we have something to sample from
        for _ in range(3):
            for i in range(50):
                done = i == 49
                self.buffer.add_step({"action": i, "reward": 1.0, "done": done})

        loader = self.buffer.sample_experiences()
        batch = next(iter(loader))  # get one batch

        self.assertIn("action", batch)
        self.assertIn("reward", batch)
        self.assertEqual(
            batch["action"].shape, (self.buffer.batch_size, self.buffer.sequence_length)
        )
        self.assertEqual(
            batch["reward"].shape, (self.buffer.batch_size, self.buffer.sequence_length)
        )
        self.assertIsInstance(batch["action"], torch.Tensor)
        self.assertIsInstance(batch["reward"], torch.Tensor)


class TestExperienceDataset(unittest.TestCase):
    def test_experience_dataset(self):
        # Mock generator that yields deterministic chunks
        def generator():
            chunk = {
                "action": np.arange(50, dtype=np.int32),
                "reward": np.ones(50, dtype=np.float32),
            }
            while True:
                yield chunk

        ds = ExperienceDataset(generator)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=10, collate_fn=collate_fn, num_workers=0
        )
        batch = next(iter(loader))

        self.assertEqual(batch["action"].shape, (10, 50))
        self.assertTrue(torch.all(batch["reward"] == 1.0))


if __name__ == "__main__":
    unittest.main()
