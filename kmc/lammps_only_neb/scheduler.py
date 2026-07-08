from __future__ import annotations

from typing import Sequence


def group_id_for_rank(rank: int, replicas_per_neb: int) -> int:
    return int(rank) // int(replicas_per_neb)


def slots_for_group(slots: Sequence[int], group_id: int, n_groups: int) -> list[int]:
    return [int(slot) for slot in slots if int(slot) % int(n_groups) == int(group_id)]


def split_group_comm(world_comm, replicas_per_neb: int):
    """Split MPI_COMM_WORLD into fixed-size replica groups."""

    rank = world_comm.Get_rank()
    size = world_comm.Get_size()
    replicas = int(replicas_per_neb)
    if replicas <= 0:
        raise ValueError("replicas_per_neb must be positive")
    if size % replicas != 0:
        raise ValueError(f"MPI size {size} is not divisible by replicas_per_neb={replicas}")
    group_id = group_id_for_rank(rank, replicas)
    return group_id, size // replicas, world_comm.Split(color=group_id, key=rank)
