import streamlit  as st
import pandas as pd

import GBM


uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

processed = GBM.preprocessing(uploaded_file)
infer = GBM.infer(processed)


st.write(processed)

st.write(infer)

if st.button("Push to train Model"):

    GBM.preprocessing()