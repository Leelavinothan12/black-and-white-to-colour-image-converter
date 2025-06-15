import streamlit as st
from PIL import Image
import torchvision.transforms as transforms
import torch
import os
import torch.nn as nn

class ColorizationNet(nn.Module):
    def __init__(self):
        super(ColorizationNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=4, dilation=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=4, dilation=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=4, dilation=2)
        self.conv4 = nn.Conv2d(128, 3, kernel_size=5, stride=1, padding=4, dilation=2)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = torch.sigmoid(self.conv4(x))
        return x
def load_model(model_path):
    model = ColorizationNet()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


model_path = os.path.abspath('./colorization_model.pth')
model = load_model(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    st.title('Image Colorization App')
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
    
        original_image = Image.open(uploaded_file)
        st.image(original_image, caption='Original Image', use_column_width=True)

        gray_image = original_image.convert("L")
        
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        img_tensor = transform(gray_image).unsqueeze(0)
        
        img_tensor = img_tensor.to(device)
        

        model.eval()
        with torch.no_grad():
            colorized_tensor = model(img_tensor)
        

        colorized_image = transforms.ToPILImage()(colorized_tensor.squeeze(0).cpu())
        

        st.image(colorized_image, caption='Colorized Image', use_column_width=True)
        


if __name__ == "__main__":
    main()
