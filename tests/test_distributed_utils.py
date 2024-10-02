import unittest
import torch
import os
from utils.distributed_utils import init_distributed_mode, cleanup

class TestDistributedUtils(unittest.TestCase):
    def test_single_process_mode(self):
        distributed_config = {}
        init_distributed_mode(distributed_config)
        self.assertFalse(distributed_config['distributed'])
        self.assertEqual(distributed_config['rank'], 0)
        self.assertEqual(distributed_config['world_size'], 1)
        self.assertEqual(distributed_config['local_rank'], 0)
        cleanup()

    def test_distributed_mode_initialization(self):
        # Simulate environment variables for distributed training
        os.environ['RANK'] = '1'
        os.environ['WORLD_SIZE'] = '4'
        os.environ['LOCAL_RANK'] = '1'

        distributed_config = {'backend': 'gloo'}
        init_distributed_mode(distributed_config)
        self.assertTrue(distributed_config['distributed'])
        self.assertEqual(distributed_config['rank'], 1)
        self.assertEqual(distributed_config['world_size'], 4)
        self.assertEqual(distributed_config['local_rank'], 1)
        cleanup()

        # Clean up environment variables
        del os.environ['RANK']
        del os.environ['WORLD_SIZE']
        del os.environ['LOCAL_RANK']

    def test_cleanup(self):
        # Simulate initialization
        distributed_config = {'backend': 'gloo'}
        init_distributed_mode(distributed_config)
        self.assertTrue(torch.distributed.is_initialized())
        cleanup()
        self.assertFalse(torch.distributed.is_initialized())

if __name__ == '__main__':
    unittest.main()
