import unittest

from export_detector_dataset import (
    DEFAULT_CLASSES,
    region_to_yolo_row,
    yolo_bbox_from_xyxy,
)


class YoloConversionTests(unittest.TestCase):
    def test_converts_xyxy_bbox_to_normalized_yolo_values(self):
        x_center, y_center, width, height = yolo_bbox_from_xyxy(
            bbox=(10, 20, 30, 60),
            image_width=100,
            image_height=200,
        )

        self.assertAlmostEqual(x_center, 0.2)
        self.assertAlmostEqual(y_center, 0.2)
        self.assertAlmostEqual(width, 0.2)
        self.assertAlmostEqual(height, 0.2)

    def test_region_to_yolo_row_filters_unwanted_types(self):
        classes = ["handwritten"]
        printed_region = {"type": "printed", "bbox": [10, 20, 30, 60]}
        handwritten_region = {"type": "handwritten", "bbox": [10, 20, 30, 60]}

        self.assertIsNone(region_to_yolo_row(printed_region, 100, 200, classes))
        self.assertEqual(
            region_to_yolo_row(handwritten_region, 100, 200, classes),
            "0 0.200000 0.200000 0.200000 0.200000",
        )

    def test_default_classes_start_with_handwritten(self):
        self.assertEqual(DEFAULT_CLASSES[0], "handwritten")


if __name__ == "__main__":
    unittest.main()
