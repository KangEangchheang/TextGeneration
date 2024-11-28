from flask import Flask, request, jsonify, render_template
from mobilelegend.backoffmodel import BackoffModel
from mobilelegend.interpolationmodel import InterpolationModel
from mobilelegend.preload import preprocess_data, build_ngrams


tokens = preprocess_data('corpus.txt')
ngrams = build_ngrams(tokens, n=4)


app = Flask(__name__, template_folder='templates')

# Initialize models
backoff_model = BackoffModel(ngrams)
interpolation_model = InterpolationModel(ngrams)


@app.route('/')
def index():
    return render_template('./index.html')  # Render the main HTML page



@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    context = data['context'].split()
    model_type = data['model']
    
    if model_type == 'backoff':
        generated_word = backoff_model.generate(context)
    elif model_type == 'interpolation':
        generated_word = interpolation_model.generate(context)
    else:
        return jsonify({'generated_text': 'Something went wrong try again'}), 400
    
    
    return jsonify({'generated_text': generated_word}), 200




if __name__ == '__main__':
    app.run(debug=True)
