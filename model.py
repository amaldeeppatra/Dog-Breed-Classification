from transformers import AutoImageProcessor, AutoModelForImageClassification
import PIL
import requests


# image = PIL.Image.open("german.jpg")
# image = PIL.Image.open("rgergherherf.jfif")
image = PIL.Image.open("dogpic.jpg")


image_processor = AutoImageProcessor.from_pretrained("wesleyacheng/dog-breeds-multiclass-image-classification-with-vit")
model = AutoModelForImageClassification.from_pretrained("wesleyacheng/dog-breeds-multiclass-image-classification-with-vit")
inputs = image_processor(images=image, return_tensors="pt")
output = model(**inputs)
logits = output.logits
predicted_class_idx = logits.argmax(-1).item()
print("Dog breed is ", model.config.id2label[predicted_class_idx])