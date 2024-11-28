
from flask import Flask, request, jsonify, render_template
from mobilelegend.backoffmodel import BackoffModel
from mobilelegend.interpolationmodel import InterpolationModel
from mobilelegend.preload import generate_4grams, preprocess_data, split_data


# Load data and build ngrams
tokens = preprocess_data('corpus.txt')
train_data, validate_data, test_data = split_data(tokens)

# Generate the 4-gram model
ngram_model = generate_4grams(train_data)






# Initialize models
backoff_model = BackoffModel(ngram_model)
print(ngram_model)
# interpolation_model = InterpolationModel()




app = Flask(__name__, template_folder='templates')

@app.route('/')
def index():
    return render_template('./index.html')  # Render the main HTML page



@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    context = data.get('context', '').split()  # Ensure context is a list of words
    model_type = data.get('model', 'backoff')  # Default to backoff if model not specified
    
    if model_type == 'backoff':
        generated_word = backoff_model.generate(ngram_model,context,10)
    # elif model_type == 'interpolation':
    #     generated_word = interpolation_model.generate(context)
    else:
        return jsonify({'generated_text': 'Something went wrong try again'}), 400
    
    generated_text = ' '.join(context + generated_word)
    
    if(generated_word == 'Context must be exactly 3 words long for this model.'):   
        return jsonify({'generated_text': "Context must be exactly 3 words long for this model."}), 500
    
    
    
    return jsonify({'generated_text': generated_text}), 200




if __name__ == '__main__':
    app.run(debug=True)
