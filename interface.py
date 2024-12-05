import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchinfo import summary
# import the function cut_image from cut_picture.py

from cut_picture import cut_image


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.model = nn.Sequential(
            # first layer
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            # second layer
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            # third layer
            nn.Flatten(),
            nn.Linear(in_features=29*29*16, out_features=120),
            nn.ReLU(),

            # fourth layer
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),

            # output layer
            nn.Linear(in_features=84, out_features=2)
        )

    def forward(self, x):
        return self.model(x)


import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import io
import base64
import torch
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize, Normalize

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])  # Use a modern Bootstrap theme
app.title = "Waste Classification: Organic or Trash"

# Load the pre-trained model and define class names
class_names = {0: "O", 1: "R"}  # 0: Organic, 1: Trash
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LeNet()
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

# Define image transformations
transforms = Compose([
    Resize((128, 128)),
    ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Function to process and predict for a single image
def process_and_predict(image_data):
    # Decode the uploaded image
    img = Image.open(io.BytesIO(base64.b64decode(image_data.split(",")[1]))).convert("RGB")
    img = transforms(img).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)
    return class_names[predicted.item()]

# App layout
app.layout = dbc.Container(
    fluid=True,
    children=[
        dbc.Row([
            dbc.Col(
                html.H1(
                    "Waste Classification Dashboard",
                    className="text-center text-dark my-4",
                ),
                width=12
            )
        ]),
        dbc.Row([
            dbc.Col([
                dcc.Upload(
                    id="upload-image",
                    children=html.Div([
                        "Drag and Drop or ",
                        html.A("Select a File", style={"color": "#007bff", "cursor": "pointer"})
                    ]),
                    style={
                        "width": "100%",
                        "height": "80px",
                        "lineHeight": "80px",
                        "borderWidth": "2px",
                        "borderStyle": "dashed",
                        "borderRadius": "10px",
                        "textAlign": "center",
                        "margin": "10px",
                        "color": "#cccccc"
                    },
                    multiple=False  # Only allow one file at a time
                ),
                html.Div(id="output-image-upload", className="mt-3", style={"textAlign": "center"}),
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Prediction Result", className="card-title text-center"),
                        html.Div(id="prediction-output", className="text-center mt-3", style={
                            "fontSize": "20px", "fontWeight": "bold", "color": "#17a2b8"
                        })
                    ])
                ], style={"backgroundColor": "#222", "borderRadius": "10px", "padding": "20px"})
            ], width=6)
        ]),
        dbc.Row([
            dbc.Col(html.Footer(
                "Powered by PyTorch and Dash",
                className="text-center text-muted mt-4"
            ), width=12)
            


        ])
    ]
)

# Callback to display uploaded image and prediction
@app.callback(
    [Output("output-image-upload", "children"),
     Output("prediction-output", "children")],
    Input("upload-image", "contents"),
    prevent_initial_call=True
)
def update_output(contents):
    if contents is not None:
        # Display the uploaded image
        img_tag = html.Img(
            src=contents,
            style={"maxWidth": "100%", "maxHeight": "300px", "borderRadius": "10px", "boxShadow": "0px 4px 10px rgba(0,0,0,0.5)"}
        )

        # Make prediction
        try:
            prediction = process_and_predict(contents)
            prediction_text = f"Prediction: {prediction} ({'Organic' if prediction == 'O' else 'Trash'})"
        except Exception as e:
            prediction_text = f"Error processing the image: {str(e)}"

        return img_tag, prediction_text
    return None, "No image uploaded."

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
