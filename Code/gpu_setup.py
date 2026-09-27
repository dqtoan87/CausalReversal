"""Ép learner nhỏ chạy trên GPU và giới hạn luồng CPU. GPU dùng chung với job khác nên có thể còn trống dưới 2 GB,
khiến vl_models.device() rơi về CPU; mỗi MLP ở đây chỉ cần khoảng 1 GB bộ nhớ GPU."""
import torch
import vl_models

torch.set_num_threads(8)
if torch.cuda.is_available():
    vl_models.device = lambda: torch.device("cuda")
