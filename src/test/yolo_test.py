import unittest
import torch
from omegaconf import DictConfig, OmegaConf
from hydra import initialize, compose
from model.yolo.yolo import YOLO  # Adjust the import path as needed

class TestYOLOWithHydra(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize Hydra and load the configuration
        with initialize(version_base=None, config_path="path/to/your/config/folder"):
            cls.cfg = compose(config_name="config_file_name")  # Replace with your config file name
        print("Hydra configuration loaded successfully:")
        print(OmegaConf.to_yaml(cls.cfg))

    def setUp(self):
        # Initialize the YOLO model with the Hydra configuration
        self.model = YOLO(self.cfg)
        print("\nYOLO model initialized with Hydra configuration.")

    def test_initialization(self):
        # Test if the model is initialized correctly with Hydra config
        print("\nRunning test_initialization...")
        self.assertEqual(self.model.model_type, "YOLO")
        self.assertEqual(self.model.num_classes, 6)
        self.assertEqual(self.model.phi, "s")
        self.assertEqual(self.model.input_shape, [640, 640])
        self.assertEqual(self.model.confidence, 0.5)
        self.assertEqual(self.model.nms_iou, 0.45)
        self.assertIsNone(self.model.ema)
        print("Model initialization test passed.")

    def test_forward_pass_training(self):
        # Test the forward pass in training mode
        print("\nRunning test_forward_pass_training...")
        self.model.train()
        images = [torch.randn(3, 640, 640)]  # Single image in a batch
        targets = [{
            'boxes': torch.tensor([[100, 100, 200, 200]]),  # Single box in xyxy format
            'labels': torch.tensor([1])  # Single class label
        }]

        # Perform forward pass
        loss_dict = self.model(images, targets)

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
        images = [torch.randn(3, 640, 640)]  # Single image in a batch

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
        print("Detections:", detections)

if __name__ == '__main__':
    unittest.main()