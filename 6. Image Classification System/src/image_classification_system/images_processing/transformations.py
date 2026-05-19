from torchvision import transforms as t
from image_classification_system.images_processing.padding import PadToSquare

def transformation_for_mean_and_std()-> t.Compose:
    transform = t.Compose([PadToSquare(), t.Resize((1024, 1024)), t.ToTensor()])
    return transform

def transformation_pipeline(mean,std)-> t.Compose:
    transform = t.Compose([PadToSquare(), t.Resize((1024, 1024)), t.ToTensor(), t.Normalize(mean, std)])
    return transform

