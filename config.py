# config.py

TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

DATASET_COLUMNS = {
    "Boston Housing": {
        "old_cols": ["crim", "zn", "indus", "chas", "nox", "rm", "age", "dis", "rad", "tax", "ptratio", "b", "lstat", "medv"],
        "new_cols": [
            "Crime Rate", "Zoned Land", "Industrial Proportion", "River Proximity", 
            "NOX Concentration", "Rooms per Dwelling", "Age of Property", 
            "Distance to Employment Centers", "Highway Accessibility", "Property Tax Rate", 
            "Pupil-Teacher Ratio", "Black Population Proportion", "Lower Status Population", 
            "Median Value"
        ],
        "target_col": "Median Value"
    },
    "California Housing": {
        "old_cols": ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude", "medianHouseValue"],
        "new_cols": [
            "Median Income", "House Age", "Average Rooms", "Average Bedrooms", 
            "Population", "Average Occupancy", "Latitude", "Longitude", "Median House Value"
        ],
        "target_col": "Median House Value"
    },
    "Medical Insurance Costs": {
        "old_cols": ["age", "sex", "bmi", "children", "smoker", "region", "charges"],
        "new_cols": ["Age", "Sex", "BMI", "Children", "Smoker", "Region", "Charges"],
        "target_col": "Charges"
    },
    "Fish Market": {
        "old_cols": ["Species", "Weight", "Length1", "Length2", "Length3", "Height", "Width"],
        "new_cols": ["Species", "Weight", "Vertical Length", "Diagonal Length", "Cross Length", "Height", "Width"],
        "target_col": "Weight"
    },
    "Salary Data": {
        "old_cols": ["YearsExperience", "Salary"],
        "new_cols": ["Years of Experience", "Salary"],
        "target_col": "Salary"
    }
}
