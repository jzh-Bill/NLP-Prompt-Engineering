import openai
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set up your OpenAI API key
openai.api_key = "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"
# Load the test dataset
test_data = pd.read_csv('PURE_test.csv')
train_data = pd.read_csv('PURE_train.csv')


# Function to classify a document
def classify_document(document_text="", prompt = "", temperature_param=0, max_tokens_param = 4096):
  # return classification_list
  response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": prompt + " Below is a list of sentences. Please classify each one as True or False (whether it is a requirement). Respond in JSON format, like this: {\"sentence1\": True, \"sentence2\": False, ...}."},
        {"role": "user", "content": document_text}
    ],
    temperature=temperature_param,
    max_tokens=max_tokens_param
  )
  # Extract and parse the JSON response
  try:
    import json
    classification_dict = json.loads(response['choices'][0]['message']['content'].strip())
    # Extract True/False values into a list
    classification_list = list(classification_dict.values())
    return classification_list
  
  except json.JSONDecodeError as e:
      print(f"Error parsing response: {e}")
      return {}

def get_sentences(data, column_name="Requirement"):
  try:
      # Load the data
      df = data

      # Check if the column exists
      if column_name not in df.columns:
          raise ValueError(f"Column {column_name} not found in the file.")
      
      # Extract the sentences into a list
      sentence_list = df[column_name].dropna().astype(str).tolist()
      return str(sentence_list)

  except Exception as e:
      return f"Error: {e}"

def get_true_label(file, column_name = "Req/Not Req"):
  try:
    # Load the CSV file
    df = file
    # Check if the column exists
    if column_name not in df.columns:
        raise ValueError(f"Column {column_name} not found in the file.")
      
    # Extract and convert labels
    label_list = df[column_name].apply(lambda x: True if str(x).strip().lower() == "req" else False).tolist()
    return label_list
  
  except Exception as e:
    return f"Error: {e}"
  

def evaluation(y_true, y_pred):
    """
    Evaluates the classification performance by calculating accuracy, precision, recall, and F1 score.

    Args:
        y_true (iterable): The true labels.
        y_pred (iterable): The predicted labels.
        positive_label (str): The label considered as positive for precision/recall/F1 metrics. Default is 'requirement'.

    Returns:
        dict: A dictionary containing accuracy, precision, recall, and F1 score.
    """
    # Convert true labels to lowercase for consistency
    if len(y_true) == 0 or len(y_pred) == 0:
      return "The size of input labels is incorrect."
    elif len(y_true) != len(y_pred):
      return "The size of true labels and predicted labels are different."

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')

    # Print the evaluation results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Return results as a dictionary
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

if __name__ == "__main__":
  prompt = ""
  document_text = get_sentences(train_data)
  print(document_text)
  
  prediction_list = classify_document(document_text, prompt)
  print("The predictions are:", prediction_list)
  
  ground_truth_label = get_true_label(train_data)
  print("The ground Truth labels are:", ground_truth_label)
  
  evaluation(ground_truth_label, prediction_list)