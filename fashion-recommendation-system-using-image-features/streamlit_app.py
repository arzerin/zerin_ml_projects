from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import streamlit as st
from PIL import Image
from scipy.spatial.distance import cosine
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image


PROJECT_DIR = Path(__file__).resolve().parent
IMAGE_DIR = PROJECT_DIR / "women-fashions" / "women fashion"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@st.cache_resource
def load_feature_model():
    base_model = VGG16(weights="imagenet", include_top=False)
    return Model(inputs=base_model.input, outputs=base_model.output)


def get_image_paths():
    return sorted(
        path
        for path in IMAGE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)


def extract_features(model, image_path):
    preprocessed_img = preprocess_image(image_path)
    features = model.predict(preprocessed_img, verbose=0).flatten()
    norm = np.linalg.norm(features)
    if norm == 0:
        return features
    return features / norm


@st.cache_data(show_spinner="Extracting image features...")
def build_feature_index(image_paths_as_strings):
    model = load_feature_model()
    image_paths = [Path(path) for path in image_paths_as_strings]
    features = [extract_features(model, path) for path in image_paths]
    return image_paths, features


def same_image(path_a, path_b):
    return path_a.resolve() == path_b.resolve() or path_a.name.strip().lower() == path_b.name.strip().lower()


def recommend_fashion_items(input_image_path, all_features, all_image_paths, model, top_n=5):
    input_features = extract_features(model, input_image_path)

    scored_items = []
    seen_names = {input_image_path.name.strip().lower()}

    for image_path, feature in zip(all_image_paths, all_features):
        image_name = image_path.name.strip().lower()

        if same_image(input_image_path, image_path):
            continue

        if image_name in seen_names:
            continue

        seen_names.add(image_name)
        similarity = 1 - cosine(input_features, feature)
        scored_items.append((similarity, image_path))

    scored_items.sort(key=lambda item: item[0], reverse=True)
    return scored_items[:top_n]


st.set_page_config(page_title="Fashion Image Recommender", layout="wide")

st.title("Fashion Image Recommender")

if not IMAGE_DIR.exists():
    st.error(f"Image folder not found: {IMAGE_DIR}")
    st.stop()

image_paths = get_image_paths()

if not image_paths:
    st.error(f"No images found in: {IMAGE_DIR}")
    st.stop()

model = load_feature_model()
all_image_paths, all_features = build_feature_index(tuple(str(path) for path in image_paths))

with st.sidebar:
    top_n = st.slider("Recommendations", min_value=1, max_value=10, value=4)
    mode = st.radio("Input type", ["Choose from dataset", "Upload image"])

input_image_path = None
temp_file = None

if mode == "Choose from dataset":
    selected_name = st.selectbox("Input image", [path.name for path in image_paths])
    input_image_path = next(path for path in image_paths if path.name == selected_name)
else:
    uploaded_file = st.file_uploader("Upload a fashion image", type=sorted(ext.lstrip(".") for ext in IMAGE_EXTENSIONS))
    if uploaded_file is not None:
        suffix = Path(uploaded_file.name).suffix
        temp_file = NamedTemporaryFile(delete=False, suffix=suffix)
        temp_file.write(uploaded_file.getbuffer())
        temp_file.close()
        input_image_path = Path(temp_file.name)

if input_image_path is None:
    st.info("Choose or upload an image to see recommendations.")
    st.stop()

recommendations = recommend_fashion_items(
    input_image_path=input_image_path,
    all_features=all_features,
    all_image_paths=all_image_paths,
    model=model,
    top_n=top_n,
)

input_col, rec_col = st.columns([1, 3])

with input_col:
    st.subheader("Input Image")
    st.image(Image.open(input_image_path), use_column_width=True)
    st.caption(input_image_path.name)

with rec_col:
    st.subheader("Recommendations")
    if not recommendations:
        st.warning("No recommendations found.")
    else:
        columns = st.columns(min(top_n, len(recommendations)))
        for column, (score, image_path) in zip(columns, recommendations):
            with column:
                if not image_path.exists():
                    st.write(f"More suggested Type: {image_path.name}")
                    continue

                st.image(Image.open(image_path), use_column_width=True)
                st.caption(f"{image_path.name}\nSimilarity: {score:.3f}")
