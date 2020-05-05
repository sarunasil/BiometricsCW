import torch
model = torch.hub.load('pytorch/vision:v0.5.0', 'deeplabv3_resnet101', pretrained=True)
model.eval()

from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from os import listdir
from os.path import join

def get_training_image_names(orientation = 'f'):# 's' - side, 'f' - front view
    training_folder = "./CW_data/test"

    filenames = []
    for file in listdir(training_folder):
        #if orientation in file:
        filenames.append(file)

    return [ join(training_folder, filename) for filename in filenames]

print (get_training_image_names())
for filename in get_training_image_names():
    print (filename)
    #filename = f"CW_data/test/DSC00186.JPG"
    input_image = Image.open(filename)
    tmp = input_image
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)

    # create a color pallette, selecting a color for each class
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    # plot the semantic segmentation predictions of 21 classes in each color
    r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
    r.putpalette(colors)

    fig, axs = plt.subplots(1,2, figsize=(50,100))
    axs[0].imshow(tmp)
    axs[1].imshow(r)

    plt.show()