import os

def inject_custom_css(st):
    """
    Injects the custom CSS into the Streamlit app.
    Call this after st.set_page_config() in every page.
    """
    css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'style.css')
    try:
        with open(css_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        pass
