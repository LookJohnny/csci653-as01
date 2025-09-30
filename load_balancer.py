"""
Dynamic Load Balancer for High-Throughput Distributed Systems

Allocates resources based on demand forecasts to handle traffic spikes.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import pandas as pd
import numpy as np
import json


logger = logging.getLogger(__name__)


@dataclass
class ResourceAllocation:
    """Represents resource allocation for a product/service."""
    product_id: str
    timestamp: datetime
    predicted_demand: float
    current_instances: int
    target_instances: int
    cpu_allocation: float  # Percentage
    memory_allocation: float  # GB
    is_hot_seller: bool
    confidence_score: float


@dataclass
class LoadBalancerState:
    """Current state of the load balancer."""
    allocations: Dict[str, ResourceAllocation] = field(default_factory=dict)
    total_instances: int = 0
    total_cpu: float = 0.0
    total_memory: float = 0.0
    last_update: Optional[datetime] = None
    history: List[Dict] = field(default_factory=list)


class DynamicLoadBalancer:
    """
    Dynamic load balancer that adjusts resource allocation based on
    demand forecasts from the ML models.
    """

    def __init__(
        self,
        min_instances: int = 1,
        max_instances: int = 10,
        hot_seller_threshold: float = 0.5,
        demand_spike_multiplier: float = 2.0,
        scale_up_threshold: float = 0.7,
        scale_down_threshold: float = 0.3,
    ):
        """
        Initialize the load balancer.

        Args:
            min_instances: Minimum instances per product
            max_instances: Maximum instances per product
            hot_seller_threshold: Probability threshold for hot seller classification
            demand_spike_multiplier: Multiplier for expected demand spike
            scale_up_threshold: CPU utilization threshold to scale up
            scale_down_threshold: CPU utilization threshold to scale down
        """
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.hot_seller_threshold = hot_seller_threshold
        self.demand_spike_multiplier = demand_spike_multiplier
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold

        self.state = LoadBalancerState()
        logger.info("Load balancer initialized")

    def load_forecasts(self, forecast_path: str) -> pd.DataFrame:
        """
        Load demand forecasts from CSV file.

        Args:
            forecast_path: Path to forecast CSV file

        Returns:
            DataFrame with forecasts
        """
        df = pd.read_csv(forecast_path)
        logger.info(f"Loaded {len(df)} forecast entries from {forecast_path}")
        return df

    def compute_resource_allocation(
        self,
        product_id: str,
        predicted_demand: float,
        is_hot_seller: bool,
        confidence: float,
        baseline_demand: float = 100.0,
    ) -> ResourceAllocation:
        """
        Compute resource allocation for a product based on predicted demand.

        Args:
            product_id: Product identifier
            predicted_demand: Predicted demand value
            is_hot_seller: Whether product is predicted to be hot seller
            confidence: Model confidence score
            baseline_demand: Baseline demand for normalization

        Returns:
            ResourceAllocation object
        """
        # Calculate demand ratio
        demand_ratio = predicted_demand / max(baseline_demand, 1.0)

        # Apply hot seller multiplier
        if is_hot_seller:
            demand_ratio *= self.demand_spike_multiplier

        # Compute target instances (logarithmic scaling for efficiency)
        target_instances = int(
            np.clip(
                self.min_instances + np.log1p(demand_ratio) * 2,
                self.min_instances,
                self.max_instances,
            )
        )

        # Allocate CPU and memory based on instances
        cpu_per_instance = 1.0  # 1 vCPU per instance
        memory_per_instance = 2.0  # 2GB per instance

        cpu_allocation = target_instances * cpu_per_instance
        memory_allocation = target_instances * memory_per_instance

        # Get current instances (or default to min)
        current_instances = self.state.allocations.get(product_id)
        current_instances = (
            current_instances.current_instances if current_instances else self.min_instances
        )

        allocation = ResourceAllocation(
            product_id=product_id,
            timestamp=datetime.now(),
            predicted_demand=predicted_demand,
            current_instances=current_instances,
            target_instances=target_instances,
            cpu_allocation=cpu_allocation,
            memory_allocation=memory_allocation,
            is_hot_seller=is_hot_seller,
            confidence_score=confidence,
        )

        return allocation

    def update_allocations(self, forecasts_df: pd.DataFrame) -> List[ResourceAllocation]:
        """
        Update resource allocations based on new forecasts.

        Args:
            forecasts_df: DataFrame with columns: series_id, yhat_blend, etc.

        Returns:
            List of updated allocations
        """
        allocations = []

        # Group by product and compute aggregate predictions
        for product_id, group in forecasts_df.groupby("series_id"):
            # Aggregate predictions (sum over forecast horizon)
            predicted_demand = group["yhat_blend"].sum()

            # Check if hot seller (if column exists)
            is_hot_seller = False
            if "is_hot_seller" in group.columns:
                is_hot_seller = group["is_hot_seller"].any()
            elif "yhat_blend" in group.columns:
                # Use high demand as proxy for hot seller
                is_hot_seller = predicted_demand > group["yhat_blend"].quantile(0.95)

            # Confidence score (if available)
            confidence = 0.8  # Default
            if "confidence" in group.columns:
                confidence = group["confidence"].mean()

            # Compute allocation
            allocation = self.compute_resource_allocation(
                product_id=str(product_id),
                predicted_demand=predicted_demand,
                is_hot_seller=is_hot_seller,
                confidence=confidence,
            )

            allocations.append(allocation)
            self.state.allocations[product_id] = allocation

        # Update state totals
        self.state.total_instances = sum(a.target_instances for a in allocations)
        self.state.total_cpu = sum(a.cpu_allocation for a in allocations)
        self.state.total_memory = sum(a.memory_allocation for a in allocations)
        self.state.last_update = datetime.now()

        # Record in history
        self.state.history.append(
            {
                "timestamp": self.state.last_update.isoformat(),
                "total_instances": self.state.total_instances,
                "total_cpu": self.state.total_cpu,
                "total_memory": self.state.total_memory,
                "num_products": len(allocations),
                "num_hot_sellers": sum(a.is_hot_seller for a in allocations),
            }
        )

        logger.info(
            f"Updated allocations for {len(allocations)} products. "
            f"Total instances: {self.state.total_instances}, "
            f"Hot sellers: {sum(a.is_hot_seller for a in allocations)}"
        )

        return allocations

    def get_scaling_actions(self) -> List[Dict]:
        """
        Generate scaling actions based on current allocations.

        Returns:
            List of scaling actions to execute
        """
        actions = []

        for product_id, allocation in self.state.allocations.items():
            if allocation.current_instances < allocation.target_instances:
                action = {
                    "action": "scale_up",
                    "product_id": product_id,
                    "from": allocation.current_instances,
                    "to": allocation.target_instances,
                    "delta": allocation.target_instances - allocation.current_instances,
                    "reason": "predicted_demand_increase",
                    "is_hot_seller": allocation.is_hot_seller,
                }
                actions.append(action)

            elif allocation.current_instances > allocation.target_instances:
                action = {
                    "action": "scale_down",
                    "product_id": product_id,
                    "from": allocation.current_instances,
                    "to": allocation.target_instances,
                    "delta": allocation.current_instances - allocation.target_instances,
                    "reason": "predicted_demand_decrease",
                    "is_hot_seller": allocation.is_hot_seller,
                }
                actions.append(action)

        logger.info(f"Generated {len(actions)} scaling actions")
        return actions

    def export_allocations(self, output_path: str) -> None:
        """
        Export current allocations to JSON file.

        Args:
            output_path: Path to output JSON file
        """
        data = {
            "timestamp": self.state.last_update.isoformat() if self.state.last_update else None,
            "total_instances": self.state.total_instances,
            "total_cpu": self.state.total_cpu,
            "total_memory": self.state.total_memory,
            "allocations": [
                {
                    "product_id": a.product_id,
                    "predicted_demand": a.predicted_demand,
                    "current_instances": a.current_instances,
                    "target_instances": a.target_instances,
                    "cpu_allocation": a.cpu_allocation,
                    "memory_allocation": a.memory_allocation,
                    "is_hot_seller": a.is_hot_seller,
                    "confidence_score": a.confidence_score,
                }
                for a in self.state.allocations.values()
            ],
            "history": self.state.history[-100:],  # Last 100 entries
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported allocations to {output_path}")

    def generate_report(self) -> str:
        """
        Generate human-readable report of current state.

        Returns:
            Report string
        """
        report = []
        report.append("=" * 60)
        report.append("DarkHorse Load Balancer Status Report")
        report.append("=" * 60)
        report.append(f"Last Update: {self.state.last_update}")
        report.append(f"Total Products: {len(self.state.allocations)}")
        report.append(f"Total Instances: {self.state.total_instances}")
        report.append(f"Total CPU: {self.state.total_cpu:.2f} vCPUs")
        report.append(f"Total Memory: {self.state.total_memory:.2f} GB")
        report.append("")

        # Hot sellers
        hot_sellers = [a for a in self.state.allocations.values() if a.is_hot_seller]
        report.append(f"Hot Sellers: {len(hot_sellers)}")

        if hot_sellers:
            report.append("\nTop 10 Hot Sellers:")
            sorted_hot = sorted(hot_sellers, key=lambda x: x.predicted_demand, reverse=True)[:10]
            for i, alloc in enumerate(sorted_hot, 1):
                report.append(
                    f"  {i}. {alloc.product_id}: "
                    f"Demand={alloc.predicted_demand:.1f}, "
                    f"Instances={alloc.target_instances}"
                )

        report.append("")
        report.append("=" * 60)

        return "\n".join(report)


def main():
    """Example usage of the load balancer."""
    import argparse

    parser = argparse.ArgumentParser(description="DarkHorse Dynamic Load Balancer")
    parser.add_argument("--forecasts", required=True, help="Path to forecast CSV")
    parser.add_argument("--output", default="load_balancer_allocations.json", help="Output JSON path")
    parser.add_argument("--min-instances", type=int, default=1, help="Minimum instances per product")
    parser.add_argument("--max-instances", type=int, default=10, help="Maximum instances per product")

    args = parser.parse_args()

    # Initialize load balancer
    lb = DynamicLoadBalancer(
        min_instances=args.min_instances,
        max_instances=args.max_instances,
    )

    # Load forecasts
    forecasts = lb.load_forecasts(args.forecasts)

    # Update allocations
    allocations = lb.update_allocations(forecasts)

    # Get scaling actions
    actions = lb.get_scaling_actions()

    # Export results
    lb.export_allocations(args.output)

    # Print report
    print(lb.generate_report())

    if actions:
        print(f"\n{len(actions)} Scaling Actions Required:")
        for action in actions[:10]:  # Show first 10
            print(f"  - {action['action'].upper()} {action['product_id']}: "
                  f"{action['from']} -> {action['to']} instances")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()