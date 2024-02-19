from flask import Flask, render_template, request
from transformers import BartTokenizer, BartForConditionalGeneration

app = Flask(__name__)

# Load the model and tokenizer
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    if request.method == 'POST':
        text = request.form['text']
        
        # Tokenize and encode the text
        inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
        
        # Generate the summary
        summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        return render_template('index.html', original_text=text, summary=summary)

if __name__ == '__main__':
    app.run(debug=True)
    
    
    # Artificial Intelligence and machine learning are the cornerstones of the next revolution in computing. These technologies hinge on the \ ability to recognize patterns then, based on data observed in the past, predict future outcomes. This explains the suggestions, Amazon offers as you shop online or how Netflix knows your penchant for bad 80s movies. Although machines utilizing AI principles are often referred to as smart,” most of these systems don’t learn on their own; the intervention of human programming is necessary. Data scientists prepare the inputs, selecting the variables to be used for predictive analytics. Deep learning, on the other hand, can do this job automatically.
