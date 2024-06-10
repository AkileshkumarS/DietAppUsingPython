import streamlit as st

def app():
    st.markdown('<h1 style="color: #F77591;">Welcome to the PCOS Classification and Diet Recommendation System</h1>', unsafe_allow_html=True)

    st.write("""
    Polycystic Ovary Syndrome (PCOS) is a common health condition that affects women and those assigned female at birth. 
    It's characterized by hormonal imbalance and metabolism problems that can impact overall health and appearance.
    """)

    # Adding some styling to section headers
    st.markdown("<h2 style='text-align: center; color: #F77591;'>Symptoms of PCOS</h2>", unsafe_allow_html=True)
    with st.expander(""):
        st.image("/content/sym.jpeg", caption="Symptoms of PCOS", use_column_width=True)

    st.markdown("<h2 style='text-align: center; color: #F77591;'>Effects of PCOS</h2>", unsafe_allow_html=True)
    with st.expander(""):
        st.image("/content/effects.jpeg", caption="Effects of PCOS", use_column_width=True)

    st.markdown("<h2 style='text-align: center; color: #F77591;'>PCOS Detection Rate around the World</h2>", unsafe_allow_html=True)
    with st.expander(""):
        st.image("/content/rate.jpeg", caption="PCOS Detection Rate", use_column_width=True)

    st.write("To get personalized diet recommendations and understand how PCOS might be affecting you, please navigate to PCOS Diet Recommendation section.")

if __name__ == "__main__":
    app()
