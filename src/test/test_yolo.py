import unittest
import torch
from omegaconf import DictConfig, OmegaConf
from hydra import initialize, compose
from model.yolo.yolo import YOLO  # Adjust the import path as needed

class TestYOLOWithHydra(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize Hydra and load the configuration
        with initialize(version_base=None, config_path="../../config"):
            cls.cfg = compose(config_name="config")  # Replace with your config file name
        print("Hydra configuration loaded successfully:")
        print(OmegaConf.to_yaml(cls.cfg))

    def setUp(self):
        # Initialize the YOLO model with the Hydra configuration
        self.model = YOLO(self.cfg.model)
        print("\nYOLO model initialized with Hydra configuration.")

    def test_initialization(self):
        # Test if the model is initialized correctly with Hydra config
        print("\nRunning test_initialization...")
        self.assertEqual(self.model.model_type, "YOLO")
        print("Model initialization test passed.")

    def test_forward_pass_training(self):
        # Test the forward pass in training mode
        print("\nRunning test_forward_pass_training...")
        self.model.train()
        images = torch.randn(1, 3, 640, 640)  # Batch size of 1

        targets = [{
            'boxes': torch.tensor([[100, 100, 200, 200]]),  # Single box in xyxy format
            'labels': torch.tensor([1])  # Single class label
        }]

        # Perform forward pass
        outputs, loss_dict = self.model(images, targets)

        # Check if the loss dictionary is returned
        self.assertIn('loss_box', loss_dict)
        self.assertIn('loss_cls', loss_dict)
        self.assertIn('loss_dfl', loss_dict)
        print("Training forward pass test passed.")
        print("Loss dictionary:", loss_dict)

    def test_forward_pass_inference(self):
        # Test the forward pass in inference mode
        print("\nRunning test_forward_pass_inference...")
        self.model.eval()
        images = torch.randn(1, 3, 640, 640)  # Batch size of 1

        # Perform forward pass
        detections = self.model(images)

        # Check if the detections are returned
        self.assertIsInstance(detections, list)
        self.assertEqual(len(detections), 1)  # One image in the batch

        # Check the structure of the detections
        detection = detections[0]
        self.assertIn('boxes', detection)
        self.assertIn('scores', detection)
        self.assertIn('labels', detection)
        print("Inference forward pass test passed.")
        print("Detections:", detections[0])

    def test_forward_pass_training_batch(self):
        # Test the forward pass in training mode with batch size > 1
        print("\nRunning test_forward_pass_training_batch...")
        self.model.train()
        batch_size = 2
        images = torch.randn(batch_size, 3, 640, 640)  # Batch size of 2

        targets = [
            {
                'boxes': torch.tensor([[100, 100, 200, 200]]),
                'labels': torch.tensor([1])
            },
            {
                'boxes': torch.tensor([[50, 50, 150, 150]]),
                'labels': torch.tensor([2])
            }
        ]

        # Perform forward pass
        outputs, loss_dict = self.model(images, targets)

        # Check if the loss dictionary is returned
        self.assertIn('loss_box', loss_dict)
        self.assertIn('loss_cls', loss_dict)
        self.assertIn('loss_dfl', loss_dict)
        print("Training forward pass with batch size test passed.")
        print("Loss dictionary:", loss_dict)

    def test_forward_pass_inference_batch(self):
        # Test the forward pass in inference mode with batch size > 1
        print("\nRunning test_forward_pass_inference_batch...")
        self.model.eval()
        batch_size = 4
        images = torch.randn(batch_size, 3, 640, 640)  # Batch size of 4

        # Perform forward pass
        detections = self.model(images)

        # Check if the detections are returned
        self.assertIsInstance(detections, list)
        self.assertEqual(len(detections), batch_size)  # Number of outputs should match batch size

        for i in range(batch_size):
            detection = detections[i]
            self.assertIn('boxes', detection)
            self.assertIn('scores', detection)
            self.assertIn('labels', detection)
            print(f"Inference forward pass test passed for batch index {i}.")
            print(f"Detections {i}:", detection)

if __name__ == '__main__':
    unittest.main()
