from transformers.trainer_utils import is_main_process
import torch.distributed as dist
import torch

def main() -> None:
    dist.init_process_group("nccl")

    
    rank = dist.get_rank()
    device = torch.device(rank)
    if is_main_process(rank):
        tensor = torch.zeros((2, 100, 100), device=device)
    else:
        tensor = torch.ones((2, 100, 70), device=device)


    output_tensors = [tensor.clone() for _ in range(dist.get_world_size())]
    dist.all_gather(output_tensors, tensor)


    print

if "__main__" in __name__:
    main()