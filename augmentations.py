
from monai.transforms import (
    EnsureChannelFirstd, Compose, LoadImaged, 
    Resized, ToTensord, ConcatItemsd, 
    NormalizeIntensityd, Orientationd
)
from monai.transforms import (
    EnsureChannelFirstd, Compose, LoadImaged, 
    Resized, ToTensord, RandGaussianNoised, RandAffined, 
    RandFlipd, ConcatItemsd, NormalizeIntensityd, 
    RandScaleIntensityd, RandShiftIntensityd, Orientationd
)

def get_test_transforms():
    return Compose([
        LoadImaged(keys=["t2w", "adc", "hbv", "seg"]),
        EnsureChannelFirstd(keys=["t2w", "adc", "hbv", "seg"]),
        ConcatItemsd(keys=["t2w", "adc", "hbv"], name="img"),
        Resized(keys=["img", "seg"], spatial_size=(160, 128, 24)),
        Orientationd(keys=["img", "seg"], axcodes="RAS"),
        NormalizeIntensityd(keys="img", nonzero=True, channel_wise=True),
        ToTensord(keys=["img", "seg"]),
    ])




def get_train_transforms():
    return Compose([
        LoadImaged(keys=["t2w", "adc", "hbv", "seg"]),
        EnsureChannelFirstd(keys=["t2w", "adc", "hbv", "seg"]),
        ConcatItemsd(keys=["t2w", "adc", "hbv"], name="img"),
        Resized(keys=["img", "seg"], spatial_size=(160, 128, 24)),
        RandAffined(keys=["img", "seg"], prob=0.2, translate_range=10.0),
        Orientationd(keys=["img", "seg"], axcodes="RAS"),
        RandFlipd(keys=["img", "seg"], spatial_axis=[0], prob=0.5),
        RandFlipd(keys=["img", "seg"], spatial_axis=[1], prob=0.5),
        RandFlipd(keys=["img", "seg"], spatial_axis=[2], prob=0.5),
        RandGaussianNoised(keys="img", prob=0.4),
        NormalizeIntensityd(keys="img", nonzero=True, channel_wise=True),
        RandScaleIntensityd(keys="img", factors=0.1, prob=0.4),
        RandShiftIntensityd(keys="img", offsets=0.1, prob=0.4),
        ToTensord(keys=["img", "seg"]),
    ])