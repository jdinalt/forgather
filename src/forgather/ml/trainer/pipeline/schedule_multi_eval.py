from typing import List, Tuple, Any, Dict, Optional, Union
import logging

from torch.distributed.pipelining.schedules import (
    PipelineScheduleMulti,
    _Action,
    _ComputationType,
    _format_pipeline_order,
)
from torch.distributed.pipelining.stage import _PipelineStageBase

logger = logging.getLogger(__name__)


class ScheduleMultiEval(PipelineScheduleMulti):
    """
    Multi-stage scheduler which only runs the forward pass

    We use this for our "eval" model
    """

    def __init__(
        self,
        stages: List[_PipelineStageBase],
        n_microbatches: int,
        stage_indices: List[Tuple[int, ...]],
        output_merge_spec: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
    ):
        super().__init__(
            stages=stages,
            n_microbatches=n_microbatches,
            loss_fn=None,
            output_merge_spec=output_merge_spec,
        )

        self.pipeline_order: Dict[int, List[Optional[_Action]]] = {}

        for rank in range(self.pp_group_size):
            rank_ops = self._calculate_single_rank_operations(rank, stage_indices)
            self.pipeline_order[rank] = rank_ops

        logger.debug(f"Eval Pipeline:\n{_format_pipeline_order(self.pipeline_order)}")

    def _calculate_single_rank_operations(self, rank, stage_indices):
        rank_stage_indices = stage_indices[rank]
        rank_ops: List[Optional[_Action]] = [None for _ in range(rank)]

        for stage_index in rank_stage_indices:
            rank_ops.extend(
                _Action(stage_index, _ComputationType.FORWARD, mb_index)
                for mb_index in range(self._n_microbatches)
            )

        return rank_ops
