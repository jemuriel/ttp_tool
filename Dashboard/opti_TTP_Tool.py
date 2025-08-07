import os

import streamlit as st


import streamlit as st

def main() -> None:
    # Page settings
    st.set_page_config(
        page_title="TTP Optimisation Tool",
        layout="wide",
        page_icon="🚆",
    )

    # Centered layout
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        # Better quality logo from Pacific National site
        st.image("Images/pn_loco.jpg", use_container_width=True)

        st.markdown(
            """
            <div style='text-align: center; padding-top: 20px;'>
                <h1 style='font-size: 42px; color: #003366;'>TTP Optimisation Tool</h1>
                <h4 style='color: #FFFFFF;'>🚆 Tactical Train Planning</h4>
                <p style='margin-top: 30px; font-size: 16px; color: #FFFFFF;'>👈 Use the sidebar to begin navigating through the tool.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

if __name__ == "__main__":
    main()
