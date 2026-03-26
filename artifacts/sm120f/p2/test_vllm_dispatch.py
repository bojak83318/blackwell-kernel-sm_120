import sys
import os
sys.path.append(os.path.join(os.getcwd(), "vllm-sm120"))

import unittest
from unittest.mock import patch, MagicMock
import torch

class TestMoeDispatch(unittest.TestCase):
    @patch("vllm.platforms.cuda.CudaPlatform.is_sm120f")
    @patch("vllm.model_executor.layers.fused_moe.cutlass_moe.ops")
    def test_sm120f_dispatch(self, mock_ops, mock_is_sm120f):
        from vllm.model_executor.layers.fused_moe.cutlass_moe import run_cutlass_moe_fp4
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation
        
        mock_is_sm120f.return_value = True
        
        # Create dummy tensors
        m, n, k, e = 1, 128, 128, 2
        topk = 1
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        output = torch.zeros((m, k), device=device, dtype=torch.half)
        a = torch.zeros((m, k), device=device, dtype=torch.half)
        a1_gscale = torch.ones(e, device=device, dtype=torch.float32)
        w1_fp4 = torch.zeros((e, 2*n, k//2), dtype=torch.uint8, device=device)
        w1_blockscale = torch.zeros((e, 2*n, k//16), dtype=torch.float8_e4m3fn, device=device)
        w1_alphas = torch.ones(e, device=device)
        a2_gscale = torch.ones(e, device=device)
        w2_fp4 = torch.zeros((e, k, n//2), dtype=torch.uint8, device=device)
        w2_blockscale = torch.zeros((e, k, n//16), dtype=torch.float8_e4m3fn, device=device)
        w2_alphas = torch.ones(e, device=device)
        topk_weights = torch.ones((m, topk), device=device)
        topk_ids = torch.zeros((m, topk), dtype=torch.long, device=device)
        workspace13 = torch.zeros(1024*1024, device=device)
        workspace2 = torch.zeros(1024*1024, device=device)
        
        # We need to mock scaled_fp4_experts_quant and other ops used in run_cutlass_moe_fp4
        mock_ops.get_cutlass_moe_mm_data.return_value = None
        mock_ops.shuffle_rows.side_effect = lambda x, y: x
        mock_ops.scaled_fp4_experts_quant.return_value = (torch.zeros_like(w1_fp4[0]), torch.zeros_like(w1_blockscale[0]))
        mock_ops.silu_and_mul_scaled_fp4_experts_quant.return_value = (torch.zeros_like(w1_fp4[0]), torch.zeros_like(w1_blockscale[0]))

        try:
            run_cutlass_moe_fp4(
                output, a, a1_gscale, w1_fp4, w1_blockscale, w1_alphas,
                a2_gscale, w2_fp4, w2_blockscale, w2_alphas,
                topk_weights, topk_ids, MoEActivation.SILU,
                workspace13, workspace2, m, n, k, e, torch.device(device)
            )
        except Exception as e:
            # We expect some failures due to mocking, but we want to see if the op was called
            print(f"Caught expected exception: {e}")
            
        # Check if cutlass_moe_fp4_sm120f_tma was called
        self.assertTrue(mock_ops.cutlass_moe_fp4_sm120f_tma.called)
        self.assertFalse(mock_ops.cutlass_fp4_moe_mm.called)

    @patch("vllm.platforms.cuda.CudaPlatform.is_sm120f")
    @patch("vllm.model_executor.layers.fused_moe.cutlass_moe.ops")
    def test_sm120a_dispatch(self, mock_ops, mock_is_sm120f):
        from vllm.model_executor.layers.fused_moe.cutlass_moe import run_cutlass_moe_fp4
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation
        
        mock_is_sm120f.return_value = False
        
        # Create dummy tensors (same as above)
        m, n, k, e = 1, 128, 128, 2
        topk = 1
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        output = torch.zeros((m, k), device=device, dtype=torch.half)
        a = torch.zeros((m, k), device=device, dtype=torch.half)
        a1_gscale = torch.ones(e, device=device, dtype=torch.float32)
        w1_fp4 = torch.zeros((e, 2*n, k//2), dtype=torch.uint8, device=device)
        w1_blockscale = torch.zeros((e, 2*n, k//16), dtype=torch.float8_e4m3fn, device=device)
        w1_alphas = torch.ones(e, device=device)
        a2_gscale = torch.ones(e, device=device)
        w2_fp4 = torch.zeros((e, k, n//2), dtype=torch.uint8, device=device)
        w2_blockscale = torch.zeros((e, k, n//16), dtype=torch.float8_e4m3fn, device=device)
        w2_alphas = torch.ones(e, device=device)
        topk_weights = torch.ones((m, topk), device=device)
        topk_ids = torch.zeros((m, topk), dtype=torch.long, device=device)
        workspace13 = torch.zeros(1024*1024, device=device)
        workspace2 = torch.zeros(1024*1024, device=device)
        
        mock_ops.get_cutlass_moe_mm_data.return_value = None
        mock_ops.shuffle_rows.side_effect = lambda x, y: x
        mock_ops.scaled_fp4_experts_quant.return_value = (torch.zeros_like(w1_fp4[0]), torch.zeros_like(w1_blockscale[0]))
        mock_ops.silu_and_mul_scaled_fp4_experts_quant.return_value = (torch.zeros_like(w1_fp4[0]), torch.zeros_like(w1_blockscale[0]))

        try:
            run_cutlass_moe_fp4(
                output, a, a1_gscale, w1_fp4, w1_blockscale, w1_alphas,
                a2_gscale, w2_fp4, w2_blockscale, w2_alphas,
                topk_weights, topk_ids, MoEActivation.SILU,
                workspace13, workspace2, m, n, k, e, torch.device(device)
            )
        except Exception as e:
            print(f"Caught expected exception: {e}")
            
        # Check if cutlass_fp4_moe_mm was called
        self.assertFalse(mock_ops.cutlass_moe_fp4_sm120f_tma.called)
        self.assertTrue(mock_ops.cutlass_fp4_moe_mm.called)

if __name__ == "__main__":
    unittest.main()
