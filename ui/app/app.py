import streamlit as st
from util import get_container_response

ACI_DOMAIN = "10.200.300.40"

input_ = st.text_input('Text goes here...')

if st.button("Extract entities!"):
    if input_ is not None:
        # Send out request to Azure Container Instance
        response = get_container_response(input_, ACI_DOMAIN)

        # Print response of ACI to the UI
        st.text(response.text)