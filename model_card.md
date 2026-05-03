# Model Card

## Model Details

This project trains a Random Forest classifier to predict whether a person's
income is greater than 50K per year using the cleaned Census Income dataset
provided in `data/census.csv`. The model is trained in `train_model.py` using
the helper functions in `ml/model.py` and `ml/data.py`. The trained model is
saved to `model/model.pkl`, and the fitted one-hot encoder is saved to
`model/encoder.pkl`.

## Intended Use

The model is intended for educational use in the Udacity Deploying a Scalable
ML Pipeline with FastAPI project. It demonstrates a complete machine learning
workflow: preprocessing tabular data, training a model, evaluating metrics,
saving artifacts, and serving predictions through a FastAPI endpoint.

The model should not be used to make real employment, credit, housing, or other
high-impact decisions about people.

## Training Data

The training data comes from the provided `data/census.csv` file. The data was
split into an 80 percent training set and a 20 percent test set with a fixed
random seed. The split was stratified by the `salary` label so both classes are
represented consistently in training and evaluation.

Categorical variables were one-hot encoded using scikit-learn's
`OneHotEncoder`. The target label was binarized with `LabelBinarizer`, where
`>50K` is represented as 1 and `<=50K` is represented as 0.

## Evaluation Data

The evaluation data is the held-out 20 percent test split from `data/census.csv`.
The test split was not used to fit the one-hot encoder or train the model.

## Metrics

The model was evaluated with precision, recall, and F1 score on the held-out
test set.

Overall model performance:

- Precision: 0.8009
- Recall: 0.5874
- F1: 0.6777

Slice performance was also computed for each unique value in every categorical
feature listed in the project. Those results are stored in `slice_output.txt`.

## Ethical Considerations

The dataset contains demographic and socioeconomic features such as sex, race,
education, occupation, relationship, and native country. These features can
encode historical inequities and may lead to biased predictions across groups.
The model card and slice metrics are included to make performance differences
more visible, but those checks do not remove the underlying fairness concerns.

Because the target is income, predictions may be correlated with sensitive
attributes and structural inequality. The model should be treated as a learning
artifact, not a decision system.

## Caveats and Recommendations

This model was built as a baseline for a course project. It uses a simple
train-test split and a Random Forest classifier without extensive
hyperparameter tuning. Additional work would be needed before any production
use, including stronger fairness analysis, model monitoring, validation on more
recent data, and review of whether the input features are appropriate for the
intended use case.
