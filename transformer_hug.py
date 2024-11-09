import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from transformers import pipeline

pipe = pipeline('object-detection', model='spark-ds549/detr-label-detection')
image_path = './data/kin-phinf.pstatic.net_20240812_254_1723432469337AsMpB_JPEG_1723432469323.jpeg'

image = Image.open(image_path)

results = pipe(image)

# Draw bounding boxes on the image
draw = ImageDraw.Draw(image)
for result in results:
    box = result['box']
    label = result['label']
    score = result['score']

    # Draw rectangle and label
    draw.rectangle(((box['xmin'], box['ymin']), (box['xmax'], box['ymax'])), outline='red', width=2)
    draw.text((box['xmin'], box['ymin']), f'{label}: {round(score, 3)}', fill='red')

plt.imshow(image)
plt.axis('off')
plt.show()
