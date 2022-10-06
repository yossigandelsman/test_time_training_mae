from torchvision import datasets
from typing import Optional, Callable, Tuple, Any
import torch
from data.imagenet_r import ImageFolderSafe


class ExtendedImageFolder(ImageFolderSafe):
    def __init__(self, root: str, batch_size: int = 1, steps_per_example: int = 1, minimizer = None, transform: Optional[Callable] = None, single_crop: bool = False, start_index: int = 0):
        super().__init__(root=root, transform=transform)
        self.batch_size = batch_size
        self.minimizer = minimizer
        self.steps_per_example = steps_per_example
        self.single_crop = single_crop
        self.start_index = start_index
    
    def __len__(self):
        mult = self.steps_per_example * self.batch_size
        mult *= (super().__len__() if self.minimizer is None else len(self.minimizer)) 
        return mult

    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        real_index = (index // self.steps_per_example) + self.start_index
        if self.minimizer is not None:
            real_index = self.minimizer[real_index]
        path, target = self.samples[real_index]
        sample = self.loader(path)
        if self.transform is not None and not self.single_crop:
            samples = torch.stack([self.transform(sample) for i in range(self.batch_size)], axis=0)
        elif self.transform and self.single_crop:
            s = self.transform(sample)
            samples = torch.stack([s for i in range(self.batch_size)], axis=0)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return samples, target


class ExtendedSplitImageFolder(ExtendedImageFolder):
    def __init__(self, root: str, batch_size: int = 1, steps_per_example: int = 1, split: int = 0, minimizer = None, 
                 transform: Optional[Callable] = None, single_crop: bool = False, start_index: int = 0):
        super().__init__(root=root, batch_size=batch_size, steps_per_example=steps_per_example, minimizer=minimizer, 
                         transform=transform, single_crop=single_crop, start_index=start_index)
        self.new_samples = []
        for i, sample in enumerate(self.samples):
            if i % 20 == split:
                self.new_samples.append(sample)
        self.samples = self.new_samples