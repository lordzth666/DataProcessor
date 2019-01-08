import sys
from src.config import Config
from src.TinyImageNet import ImageNet

if len(sys.argv) < 4:
    print("Usage: python DataProcessor.py [type] [source_data_path] [output_base_path]")
    exit(1)

config = Config(sys.argv)
Container = ImageNet(config)
Container.process()