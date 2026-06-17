import csv
import os
from tensorflow.keras import applications

# from custom_models import get_custom_model

# https://keras.io/api/applications/
base_models = {
    "convnext_tiny":        applications.ConvNeXtTiny,
    "convnext_small":       applications.ConvNeXtSmall,
    "convnext_base":        applications.ConvNeXtBase,
    "convnext_large":       applications.ConvNeXtLarge,
    "convnext_xlarge":      applications.ConvNeXtXLarge,
    "densenet121":          applications.DenseNet121,
    "densenet169":          applications.DenseNet169,
    "densenet201":          applications.DenseNet201,
# FIXME bug :(    "efficientnetb0":       applications.EfficientNetB0,
# FIXME bug :(    "efficientnetb1":       applications.EfficientNetB1,
# FIXME bug :(    "efficientnetb2":       applications.EfficientNetB2,
# FIXME bug :(    "efficientnetb3":       applications.EfficientNetB3,
# FIXME bug :(    "efficientnetb4":       applications.EfficientNetB4,
# FIXME bug :(    "efficientnetb5":       applications.EfficientNetB5,
# FIXME bug :(    "efficientnetb6":       applications.EfficientNetB6,
# FIXME bug :(    "efficientnetb7":       applications.EfficientNetB7,
    "efficientnetv2-b0":    applications.EfficientNetV2B0,
    "efficientnetv2-b1":    applications.EfficientNetV2B1,
    "efficientnetv2-b2":    applications.EfficientNetV2B2,
    "efficientnetv2-b3":    applications.EfficientNetV2B3,
    "inception_resnet_v2":  applications.InceptionResNetV2,
    "inception_v3":         applications.InceptionV3,
    "_mobilenet":           applications.MobileNet,
    "_mobilenetv2":         applications.MobileNetV2,
# FIXME not implemented    "MobileNetV3Small":     applications.MobileNetV3Small,
# FIXME not implemented    "MobileNetV3Large":     applications.MobileNetV3Large,
    "nasnet_large":         applications.NASNetLarge,
    "nasnet_mobile":        applications.NASNetMobile,
    "resnet50":             applications.ResNet50,
    "resnet101":            applications.ResNet101,
    "resnet152":            applications.ResNet152,
    "resnet50v2":           applications.ResNet50V2,
    "resnet101v2":          applications.ResNet101V2,
    "resnet152v2":          applications.ResNet152V2,
    "vgg16":                applications.VGG16,
    "vgg19":                applications.VGG19,
    "xception":             applications.Xception
    # "custom":               get_custom_model()
}

def available_models():
    return list(base_models.keys())

def string_to_model(modelstr):
    if not modelstr in base_models:
        raise NotImplementedError(modelstr + " is not yet implemented")

    return base_models[modelstr]

def get_model_config(model_name, models_csv=os.path.join(os.path.dirname(__file__), "..", "models.csv")):
    if not os.path.isfile(models_csv):
        raise ValueError("models_csv does not exist: " + models_csv)

    with open(models_csv, 'r') as ifile:
        rows = csv.DictReader(ifile)
        for row in rows:
            if row["name"] == model_name:
                return row.copy()

    raise ValueError("Unknown model: " + model_name)

# EOF
