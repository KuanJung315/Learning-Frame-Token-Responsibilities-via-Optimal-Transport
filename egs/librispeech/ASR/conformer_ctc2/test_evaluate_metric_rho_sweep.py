#!/usr/bin/env python3

import torch

from evaluate_metric_rho_sweep import (
    _distance_sensitivity,
    _plan_sensitivity,
    _spearman,
)


def main() -> None:
    distance = torch.tensor(
        [[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]]
    )
    same_distance = _distance_sensitivity(distance, distance)
    assert same_distance["distance_relative_fro_to_rho0"] == 0.0
    assert abs(same_distance["distance_cosine_to_rho0"] - 1.0) < 1e-6
    scaled_distance = _distance_sensitivity(distance, 2.0 * distance)
    assert abs(scaled_distance["distance_relative_fro_to_rho0"] - 1.0) < 1e-6
    assert abs(scaled_distance["distance_mean_abs_relative_to_rho0"] - 1.0) < 1e-6
    assert abs(scaled_distance["distance_cosine_to_rho0"] - 1.0) < 1e-6

    plan = torch.tensor(
        [[0.45, 0.05], [0.05, 0.40], [0.00, 0.05]], dtype=torch.float32
    )
    same_plan = _plan_sensitivity(plan, plan)
    assert same_plan["plan_tv_to_rho0"] == 0.0
    assert same_plan["plan_barycenter_mae_to_rho0"] == 0.0
    assert same_plan["plan_support_iou_to_rho0"] == 1.0

    shifted = torch.roll(plan, shifts=1, dims=0)
    shifted_plan = _plan_sensitivity(plan, shifted)
    assert shifted_plan["plan_tv_to_rho0"] > 0.0
    assert shifted_plan["plan_barycenter_mae_to_rho0"] > 0.0
    assert shifted_plan["plan_support_iou_to_rho0"] < 1.0

    assert abs(_spearman([0, 0.25, 0.5, 0.75, 1], [0, 1, 2, 3, 4]) - 1.0) < 1e-9
    assert abs(_spearman([0, 0.25, 0.5, 0.75, 1], [4, 3, 2, 1, 0]) + 1.0) < 1e-9
    assert _spearman([0, 1], [2, 2]) is None
    print("metric rho-sweep helper tests passed")


if __name__ == "__main__":
    main()
