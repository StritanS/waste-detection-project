import torch
import torchvision
from torch import nn
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from PIL import Image
import io
import base64
import os
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
from cut_picture import cut_image  # Import your custom cut_image function


# Define the LeNet model
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=29 * 29 * 16, out_features=120),
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=2),
        )

    def forward(self, x):
        return self.model(x)


# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
app.title = "Waste Classification: Organic or Trash"

# Load pre-trained model
class_names = {0: "O", 1: "R"}  # O: Organic, R: Trash
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LeNet()
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

# Define image transformations
transforms = Compose([
    Resize((128, 128)),
    ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


# Function to process and predict for segmented images
def process_and_predict_segmented(folder_path):
    predictions = []
    for img_name in sorted(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, img_name)
        img = Image.open(img_path).convert("RGB")
        img = transforms(img).unsqueeze(0).to(device)

        # Predict using the model
        with torch.no_grad():
            output = model(img)
            _, predicted = torch.max(output, 1)
            class_name = class_names[predicted.item()]
            predictions.append((img_path, class_name))
    return predictions


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
                    multiple=False
                ),
                html.Div(id="output-image-upload", className="mt-3", style={"textAlign": "center"}),
            ], width=6),
            dbc.Col([
                html.Div(id="segmented-images-output", className="text-center")
            ], width=6),
        ]),
        dbc.Row([
            dbc.Col(html.Footer(
                "Powered by PyTorch and Dash",
                className="text-center text-muted mt-4"
            ), width=12)
        ]),
    ]
)


# Callback to display uploaded image, segment it, and predict
@app.callback(
    [Output("output-image-upload", "children"),
     Output("segmented-images-output", "children")],
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

        # Save uploaded image temporarily
        uploaded_img_path = "temp_uploaded_image.jpg"
        with open(uploaded_img_path, "wb") as f:
            f.write(base64.b64decode(contents.split(",")[1]))

        # Apply cut_image function
        segmented_folder = "segmented_objects_precise"
        os.makedirs(segmented_folder, exist_ok=True)
        cut_image(uploaded_img_path)  # Segments are saved in segmented_objects_precise

        # Process and predict for each segment
        predictions = process_and_predict_segmented(segmented_folder)

        # Display segmented images and predictions
        segmented_results = []
        for img_path, class_name in predictions:
            with open(img_path, "rb") as f:
                img_base64 = base64.b64encode(f.read()).decode()
            segmented_img_tag = html.Img(
                src=f"data:image/png;base64,{img_base64}",
                style={"maxWidth": "100px", "maxHeight": "100px", "margin": "5px", "borderRadius": "5px"}
            )
            prediction_text = html.Div(
                f"{class_name} ({'Organic' if class_name == 'O' else 'Trash'})",
                style={"textAlign": "center", "color": "#17a2b8"}
            )
            segmented_results.append(html.Div([segmented_img_tag, prediction_text], style={"display": "inline-block"}))

        return img_tag, html.Div(segmented_results, style={"textAlign": "center"})
    return None, None


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
    