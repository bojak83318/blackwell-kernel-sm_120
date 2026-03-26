import sys
import os
sys.path.append(os.path.join(os.getcwd(), "vllm-sm120"))

import torch
# Mock torch.version.cuda
import unittest
from unittest.mock import patch, MagicMock
from vllm.platforms.cuda import CudaPlatform

class TestCudaDetection(unittest.TestCase):
    @patch("vllm.platforms.cuda.CudaPlatform.get_device_capability")
    @patch("torch.version.cuda", "12.9")
    def test_sm120a_detection(self, mock_get_cap):
        from vllm.platforms.interface import DeviceCapability
        mock_get_cap.return_value = DeviceCapability(major=12, minor=0)
        
        self.assertTrue(CudaPlatform.is_sm120a())
        self.assertFalse(CudaPlatform.is_sm120f())

    @patch("vllm.platforms.cuda.CudaPlatform.get_device_capability")
    @patch("torch.version.cuda", "13.0")
    def test_sm120f_detection(self, mock_get_cap):
        from vllm.platforms.interface import DeviceCapability
        mock_get_cap.return_value = DeviceCapability(major=12, minor=0)
        
        self.assertFalse(CudaPlatform.is_sm120a())
        self.assertTrue(CudaPlatform.is_sm120f())

    @patch("vllm.platforms.cuda.CudaPlatform.get_device_capability")
    @patch("torch.version.cuda", "13.1")
    def test_sm120f_detection_higher(self, mock_get_cap):
        from vllm.platforms.interface import DeviceCapability
        mock_get_cap.return_value = DeviceCapability(major=12, minor=0)
        
        self.assertFalse(CudaPlatform.is_sm120a())
        self.assertTrue(CudaPlatform.is_sm120f())

    @patch("vllm.platforms.cuda.CudaPlatform.get_device_capability")
    @patch("torch.version.cuda", "12.8")
    def test_sm120a_cuda128(self, mock_get_cap):
        from vllm.platforms.interface import DeviceCapability
        mock_get_cap.return_value = DeviceCapability(major=12, minor=0)
        self.assertTrue(CudaPlatform.is_sm120a())
        self.assertFalse(CudaPlatform.is_sm120f())

    @patch("vllm.platforms.cuda.CudaPlatform.get_device_capability")
    @patch("torch.version.cuda", "13.0")
    def test_sm120f_rtx5080(self, mock_get_cap):
        # Assuming RTX 5080 is also major 12
        from vllm.platforms.interface import DeviceCapability
        mock_get_cap.return_value = DeviceCapability(major=12, minor=0)
        self.assertFalse(CudaPlatform.is_sm120a())
        self.assertTrue(CudaPlatform.is_sm120f())

    @patch("vllm.platforms.cuda.CudaPlatform.get_device_capability")
    @patch("torch.version.cuda", "13.0")
    def test_sm120f_rtx5070ti(self, mock_get_cap):
        # Assuming RTX 5070 Ti is also major 12
        from vllm.platforms.interface import DeviceCapability
        mock_get_cap.return_value = DeviceCapability(major=12, minor=0)
        self.assertFalse(CudaPlatform.is_sm120a())
        self.assertTrue(CudaPlatform.is_sm120f())

    @patch("vllm.platforms.cuda.CudaPlatform.get_device_capability")
    @patch("torch.version.cuda", "13.0")
    def test_hopper_cuda13(self, mock_get_cap):
        from vllm.platforms.interface import DeviceCapability
        mock_get_cap.return_value = DeviceCapability(major=9, minor=0)
        self.assertFalse(CudaPlatform.is_sm120a())
        self.assertFalse(CudaPlatform.is_sm120f())

if __name__ == "__main__":
    unittest.main()
