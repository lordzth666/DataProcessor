import sys
from src.config import Config
from src.TinyImageNet import ImageNet

config = Config(sys.argv)
Container = ImageNet(config)
Container.process()