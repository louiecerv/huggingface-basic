import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Specify the pre-trained model and tokenizer (replace with your desired task)
model_name = "bert-base-uncased"  # Example for sentiment analysis
task = "sentiment-analysis"  # Adjust for your specific task

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def predict(text):
    """Performs prediction using the loaded Hugging Face Transformer model.

    Args:
        text (str): The user-provided text for analysis.

    Returns:
        str: The predicted label or output (depends on the chosen model and task).
    """

    # Preprocess the text (tokenization, padding, etc.)
    inputs = tokenizer(text, return_tensors="pt")

    # Make predictions using the model
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)

    # Map predictions to labels or interpret results based on your task
    if task == "sentiment-analysis":
        labels = ["Negative", "Neutral", "Positive"]
        return labels[predictions.item()]
    else:
        # Handle other tasks (e.g., question answering, summarization)
        # by interpreting the model's output based on the specific task
        pass
      
def app():
  st.title("Hugging Face Transformer with Streamlit")
  st.write("This app lets you interact with a pre-trained Transformer model.")
  user_text = st.text_input("Enter your text here:")
  
  if st.button("Predict"):
      if user_text:
          prediction = predict(user_text)
          st.write(f"**Prediction:** {prediction}")
      else:
          st.warning("Please enter some text to analyze.")
  
  st.info(f"**Model:** {model_name}")
  st.info(f"**Task:** {task}")

# Run the app
if __name__ == "__main__":
    app()
