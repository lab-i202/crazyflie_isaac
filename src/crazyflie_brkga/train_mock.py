from cfbrkga.config import load_config
from cfbrkga.mock_evaluator import MockBatchEvaluator
from cfbrkga.training_loop import TrainingCoordinator


if __name__ == "__main__":
    config = load_config()
    evaluator = MockBatchEvaluator(config, run_output_dir=None)
    coordinator = TrainingCoordinator(config, evaluator)
    output_dir = coordinator.run()
    print(f"Mock training finished: {output_dir}")
