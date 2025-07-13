
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from streamlit.testing.v1 import AppTest
from unittest.mock import patch
from io import BytesIO

def test_app_with_mocked_uploader():
    at = AppTest.from_file("app.py")

    # The content for the mock file
    mock_file_content = b"feature1,feature2,target\n1,2,3\n2,4,6\n3,6,9\n4,8,12\n5,10,15\n"
    mock_file = BytesIO(mock_file_content)
    mock_file.name = "test.csv"

    # Mock st.file_uploader to return the mock file
    with patch("streamlit.file_uploader") as mock_uploader:
        mock_uploader.return_value = mock_file

        at.run()

        # After upload, we should see the preview and selectors
        assert len(at.multiselect) == 1
        assert len(at.selectbox) == 1
        assert at.multiselect[0].options == ["feature1", "feature2", "target"]

        # Select features and target
        at.multiselect[0].set_value(["feature1", "feature2"])
        at.selectbox[0].set_value("target")
        at.run()

        # Click the train button
        at.button(key="train_button").click()
        at.run()

        # After training, we should see the performance metrics
        assert "Model Performance" in at.markdown[1].value
        assert "Coefficients" in at.markdown[2].value
        assert "R2 Score" in at.markdown[4].value

