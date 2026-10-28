import os

def inject_custom_css(st):
    """
    Injects the Tactical HUD CSS and hardware-overlay into the Streamlit app.
    """
    css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'style.css')
    try:
        with open(css_path) as f:
            # Inject both the scanline overlay and the custom CSS
            st.markdown('<div class="hud-scanline"></div>', unsafe_allow_html=True)
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        pass
