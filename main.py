import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont
import io
import os
from streamlit_image_select import image_select

# --- Caching & Functions ---
@st.cache_data
def load_image(uploaded_file_bytes):
    image_pil = Image.open(io.BytesIO(uploaded_file_bytes))
    try:
        image_pil = ImageOps.exif_transpose(image_pil)
    except Exception:
        pass
    if image_pil.mode != 'RGB':
        image_pil = image_pil.convert('RGB')
    return np.array(image_pil)

@st.cache_data
def get_denoised_image(_img_original_np, method, h, d, sigma_color, sigma_space, ksize, nlm_template_size, nlm_search_size):
    if method == "Non-local Means":
        return cv2.fastNlMeansDenoisingColored(_img_original_np, None, float(h), float(h), nlm_template_size, nlm_search_size)
    elif method == "Bilateral Filter":
        return cv2.bilateralFilter(_img_original_np, d, sigma_color, sigma_space)
    elif method == "Gaussian Blur":
        return cv2.GaussianBlur(_img_original_np, (ksize, ksize), 0)
    elif method == "Blended NLM-Bilateral":
        nlm_result = cv2.fastNlMeansDenoisingColored(_img_original_np, None, float(h), float(h), nlm_template_size, nlm_search_size)
        return cv2.bilateralFilter(nlm_result, d, sigma_color, sigma_space)
    return _img_original_np

@st.cache_data
def apply_core_adjustments(_base_image_np, brightness, contrast, saturation, temperature, sharpness):
    adjusted_image = _base_image_np.copy().astype(np.uint8)
    if brightness != 0 or contrast != 1.0:
        adjusted_image = cv2.convertScaleAbs(adjusted_image, alpha=contrast, beta=brightness)
    if saturation != 1.0:
        hsv = cv2.cvtColor(adjusted_image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        s = np.clip(s.astype(np.float32) * saturation, 0, 255).astype(np.uint8)
        adjusted_image = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2RGB)
    if temperature != 0:
        max_delta = 40
        delta = (temperature / 50.0) * max_delta
        b, g, r = cv2.split(cv2.cvtColor(adjusted_image, cv2.COLOR_RGB2BGR))
        if delta > 0: # Warm
            r = np.clip(r.astype(np.int16) + delta, 0, 255).astype(np.uint8)
            b = np.clip(b.astype(np.int16) - delta, 0, 255).astype(np.uint8)
        else: # Cool
            r = np.clip(r.astype(np.int16) + delta, 0, 255).astype(np.uint8)
            b = np.clip(b.astype(np.int16) - delta, 0, 255).astype(np.uint8)
        adjusted_image = cv2.cvtColor(cv2.merge([b, g, r]), cv2.COLOR_BGR2RGB)
    if sharpness > 0:
        amount = sharpness / 100.0 * 1.5
        blurred = cv2.GaussianBlur(adjusted_image, (0, 0), sigmaX=1.0)
        sharpened_img = cv2.addWeighted(adjusted_image, 1.0 + amount, blurred, -amount, 0)
        adjusted_image = np.clip(sharpened_img, 0, 255).astype(np.uint8)
    return adjusted_image

# --- Filter Functions ---
def apply_moonlit_blue_filter(img_rgb):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s_adjusted = np.clip(s.astype(np.float32) * 0.65, 0, 255).astype(np.uint8)
    temp_hsv = cv2.merge((h, s_adjusted, v))
    temp_bgr = cv2.cvtColor(temp_hsv, cv2.COLOR_HSV2BGR)
    b, g, r = cv2.split(temp_bgr)
    b = np.clip(b.astype(np.int16) + 20, 0, 255).astype(np.uint8)
    g = np.clip(g.astype(np.int16) + 5, 0, 255).astype(np.uint8)
    r = np.clip(r.astype(np.int16) - 15, 0, 255).astype(np.uint8)
    filtered_bgr = cv2.merge((b, g, r))
    return cv2.cvtColor(filtered_bgr, cv2.COLOR_BGR2RGB)

def apply_warm_glow_filter(img_rgb):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    b, g, r = cv2.split(img_bgr)
    r_boost = np.clip(r.astype(np.int16) + 25, 0, 255).astype(np.uint8)
    b_damp = np.clip(b.astype(np.int16) - 15, 0, 255).astype(np.uint8)
    filtered_bgr = cv2.merge((b_damp, g, r_boost))
    return cv2.cvtColor(filtered_bgr, cv2.COLOR_BGR2RGB)

def apply_cool_night_filter(img_rgb):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    b, g, r = cv2.split(img_bgr)
    b_boost = np.clip(b.astype(np.int16) + 25, 0, 255).astype(np.uint8)
    r_damp = np.clip(r.astype(np.int16) - 15, 0, 255).astype(np.uint8)
    filtered_bgr = cv2.merge((b_boost, g, r_damp))
    return cv2.cvtColor(filtered_bgr, cv2.COLOR_BGR2RGB)

def apply_classic_bw_filter(img_rgb):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_contrast = clahe.apply(gray)
    return cv2.cvtColor(gray_contrast, cv2.COLOR_GRAY2RGB)

def apply_night_clarity_filter(img_rgb):
    kernel_sharpen = np.array([[-0.5, -0.5, -0.5], [-0.5,  5.0, -0.5], [-0.5, -0.5, -0.5]])
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    sharpened_bgr = cv2.filter2D(img_bgr, -1, kernel_sharpen)
    return cv2.cvtColor(sharpened_bgr, cv2.COLOR_BGR2RGB)

# --- Filter & Preset Definitions ---
FILTERS = {
    "Original": {"function": None, "image_path": "image/original.png"},
    "Moonlit Blue": {"function": apply_moonlit_blue_filter, "image_path": "image/moonlit_blue.png"},
    "Warm Glow": {"function": apply_warm_glow_filter, "image_path": "image/warm_glow.png"},
    "Cool Night": {"function": apply_cool_night_filter, "image_path": "image/cool_night.png"},
    "Classic B&W": {"function": apply_classic_bw_filter, "image_path": "image/classic_bw.png"},
    "Night Clarity": {"function": apply_night_clarity_filter, "image_path": "image/night_clarity.png"}
}

PRESETS = {
    "Preset 1": {
        "image_path": "image/preset_1.png",
        "settings": {
            "filter_name": "Original",
            "denoise_method": "Non-local Means",
            "h_value": 16,
            "blend_factor": 0.15,
            "brightness": 11,
            "contrast": 1.10,
            "saturation": 1.25,
            "temperature": 2,
            "sharpness": 6,
            "d_bilateral": 9,
            "sigma_color": 75,
            "sigma_space": 75,
            "ksize_gaussian": 5
        }
    },
    "Preset 2": {
        "image_path": "image/preset_2.png",
        "settings": {
            "filter_name": "Warm Glow",
            "denoise_method": "Gaussian Blur",
            "h_value": 7,
            "blend_factor": 0.15,
            "brightness": 13,
            "contrast": 0.75,
            "saturation": 1.40,
            "temperature": -15,
            "sharpness": 0,
            "d_bilateral": 9,
            "sigma_color": 75,
            "sigma_space": 75,
            "ksize_gaussian": 7
        }
    },
}

# --- Streamlit App ---
st.set_page_config(layout="wide")
st.title("Night Shoot Optimizer 🌙")

uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    uploaded_file_bytes = uploaded_file.getvalue()
    img_original_np = load_image(uploaded_file_bytes)

    st.sidebar.title("🛠️ Editing Controls")
    tab_presets, tab_denoise, tab_adjustments, tab_filter = st.sidebar.tabs(["Presets", "Denoising", "Adjustments", "Filters"])
    
    with tab_presets:
        st.subheader("🌟 Apply a Preset")
        st.write("Click a preset to apply its settings.")

        if 'last_preset' not in st.session_state:
            st.session_state.last_preset = None
            
        preset_names = list(PRESETS.keys())
        preset_images = [PRESETS[name]["image_path"] for name in preset_names]

        selected_preset_idx = image_select(
            label="Select a preset",
            images=preset_images,
            captions=preset_names,
            use_container_width=True,
            return_value='index'
        )
        
        selected_preset_name = preset_names[selected_preset_idx]

        if selected_preset_name != st.session_state.last_preset:
            st.session_state.last_preset = selected_preset_name
            preset_settings = PRESETS[selected_preset_name]["settings"]
            for key, value in preset_settings.items():
                st.session_state[key] = value
            st.success(f"Applied '{selected_preset_name}' preset!")

    with tab_denoise:
        st.subheader("⚙️ Denoising Parameters")
        denoise_method_selected = st.selectbox("Choose denoising method", ["Non-local Means", "Bilateral Filter", "Gaussian Blur", "Blended NLM-Bilateral"], key="denoise_method")
        h_value, d_bilateral, sigma_color_bilateral, sigma_space_bilateral, ksize_gaussian = 7, 9, 75, 75, 5
        nlm_template_size, nlm_search_size = 3, 11
        if denoise_method_selected in ["Non-local Means", "Blended NLM-Bilateral"]:
            h_value = st.slider("Denoising strength (NLM)", 3, 25, key="h_value")
        if denoise_method_selected in ["Bilateral Filter", "Blended NLM-Bilateral"]:
            d_bilateral = st.slider("Filter diameter (Bilateral)", 5, 15, key="d_bilateral")
            sigma_color_bilateral = st.slider("Sigma Color (Bilateral)", 50, 150, key="sigma_color")
            sigma_space_bilateral = st.slider("Sigma Space (Bilateral)", 50, 150, key="sigma_space")
        if denoise_method_selected == "Gaussian Blur":
            ksize_gaussian = st.slider("Kernel size (Gaussian)", 1, 15, 2, key="ksize_gaussian")
            if ksize_gaussian % 2 == 0: ksize_gaussian += 1
        blend_factor_val = st.slider("Naturalness level", 0.0, 1.0, key="blend_factor", help="Menggabungkan hasil denoising dengan gambar asli untuk menjaga detail alami")

    with tab_adjustments:
        st.subheader("🔧 Image Adjustments")
        brightness_val = st.slider("Brightness", -100, 100, key="brightness")
        contrast_val = st.slider("Contrast", 0.1, 3.0, key="contrast")
        saturation_val = st.slider("Color Saturation", 0.0, 3.0, key="saturation")
        temperature_val = st.slider("Color Temperature (Cool ↔ Warm)", -50, 50, key="temperature")
        sharpness_val = st.slider("Sharpness", 0, 100, key="sharpness")
    
    with tab_filter:
        st.subheader("🌃 Artistic Filters")
        st.write("Click an image to apply the filter. (A preset may change this selection).")
        
        filter_names = list(FILTERS.keys())
        preview_images = [FILTERS[name]["image_path"] for name in filter_names]

        # indeks default dari session_state
        default_index = 0
        if 'filter_name' in st.session_state:
            try:
                default_index = filter_names.index(st.session_state.filter_name)
            except ValueError:
                default_index = 0 

        selected_index = image_select(
            label="Select a filter",
            images=preview_images,
            captions=filter_names,
            use_container_width=True,
            return_value='index',
            index=default_index 
        )
        
        selected_filter_name = filter_names[selected_index]
        st.session_state.filter_name = selected_filter_name


    # --- Image Processing Pipeline ---
    denoised_image_np = get_denoised_image(img_original_np, st.session_state.denoise_method, st.session_state.h_value, st.session_state.d_bilateral, st.session_state.sigma_color, st.session_state.sigma_space, st.session_state.ksize_gaussian, nlm_template_size, nlm_search_size)
    base_image_np = cv2.addWeighted(denoised_image_np, 1.0 - st.session_state.blend_factor, img_original_np, st.session_state.blend_factor, 0)
    base_image_np = np.clip(base_image_np, 0, 255).astype(np.uint8)
    
    adjusted_image_np = apply_core_adjustments(base_image_np, st.session_state.brightness, st.session_state.contrast, st.session_state.saturation, st.session_state.temperature, st.session_state.sharpness)

    final_image_np = adjusted_image_np
    active_filter_name = st.session_state.get('filter_name', 'Original') 
    if active_filter_name != "Original":
        filter_function = FILTERS[active_filter_name]["function"]
        final_image_np = filter_function(final_image_np)
        final_image_np = np.clip(final_image_np, 0, 255).astype(np.uint8)

    # --- Show Result ---
    st.subheader("🖼️ Real-time Preview")
    st.success(f"**Active Filter:** {active_filter_name} | **Active Preset:** {st.session_state.last_preset or 'None'}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original Image**")
        st.image(img_original_np, use_container_width=True)
    with col2:
        st.markdown("**Edited Result**")
        st.image(final_image_np, use_container_width=True)
        
    st.markdown("---")
    st.header("📥 Download Result")
    result_pil_download = Image.fromarray(final_image_np)
    buf = io.BytesIO()
    result_pil_download.save(buf, format="PNG")
    download_filename = f"edited_{active_filter_name.replace(' ', '_')}_{uploaded_file.name.split('.')[0]}.png"
    st.download_button(label="📥 Download Edited Image", data=buf.getvalue(), file_name=download_filename, mime="image/png", use_container_width=True)
    
else:
    st.info("👆 Please upload an image to begin.")
    st.markdown("""
    **App Features:**
    - ✨ Real-time preview as you change parameters
    - 🌟 One-click presets for quick edits
    - 🔧 Multiple denoising methods with performance controls
    - 🎨 Basic adjustments (Brightness, Contrast, Saturation, Temperature, Sharpness)
    - 🎭 Blending with the original image
    - 🌃 A selection of artistic night-time filters
    - 📥 Download your final edit
    """)