import ydf  # Yggdrasil Decision Forests
import pandas as pd  # Used for loading and manipulating small datasets

ds_path = "https://raw.githubusercontent.com/google/yggdrasil-decision-forests/main/yggdrasil_decision_forests/test_data/dataset"

# Download and load the dataset into Pandas DataFrames
train_ds = pd.read_csv(f"{ds_path}/adult_train.csv")
test_ds = pd.read_csv(f"{ds_path}/adult_test.csv")

# Display the first 5 rows of the training data
print(train_ds.head(5))

model = ydf.GradientBoostedTreesLearner(label="income").train(train_ds)

print(model.describe())

predictions = model.predict(test_ds)

print(predictions[:5])

evaluation = model.evaluate(test_ds)

# Query individual evaluation metrics
print(f"Test accuracy: {evaluation.accuracy}")

# Show the full evaluation report
print("Full evaluation report:")
print(evaluation)
